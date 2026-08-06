#!/usr/bin/env bash
set -euo pipefail

run_dir="${1:?Usage: bash scripts/eval_search_qa_checkpoints.sh RUN_DIR [ENGINE]}"
engine="${2:-vllm}"
start_step="${START_STEP:-90}"
end_step="${END_STEP:-150}"
force_reeval="${FORCE_REEVAL:-0}"
eval_gpu_group_0="${EVAL_GPU_GROUP_0:-0,1,2,3}"
eval_gpu_group_1="${EVAL_GPU_GROUP_1:-4,5,6,7}"
eval_val_batch_size="${EVAL_VAL_BATCH_SIZE_PER_JOB:-${VAL_BATCH_SIZE:-512}}"

if [[ ! -d "$run_dir" ]]; then
  echo "Run directory not found: $run_dir" >&2
  exit 1
fi
if ! [[ "$start_step" =~ ^[0-9]+$ && "$end_step" =~ ^[0-9]+$ ]]; then
  echo "START_STEP and END_STEP must be non-negative integers." >&2
  exit 1
fi
if (( start_step > end_step )); then
  echo "START_STEP cannot be greater than END_STEP." >&2
  exit 1
fi

# Match the paths used by the SearchQA training scripts while still allowing
# callers to override every path through exported environment variables.
export MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-7B-Instruct}"
export SEARCH_RETRIEVER_PATH="${SEARCH_RETRIEVER_PATH:-models/e5-base-v2}"
if [[ -z "${SEARCH_DATA:-}" ]]; then
  if [[ -d "$PWD/data/search_data" ]]; then
    export SEARCH_DATA="$PWD/data/search_data"
  else
    export SEARCH_DATA="$PWD/search_data"
  fi
fi

output_root="$run_dir/test_best_checkpoint"
mkdir -p "$output_root"

export SEARCH_PORT="${SEARCH_PORT:-8030}"
export SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:${SEARCH_PORT}/retrieve}"
search_service_pid=""
evaluation_pids=()

cleanup_resources() {
  for pid in "${evaluation_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${evaluation_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  if [[ -n "$search_service_pid" ]] && kill -0 "$search_service_pid" 2>/dev/null; then
    kill "$search_service_pid" 2>/dev/null || true
    wait "$search_service_pid" 2>/dev/null || true
  fi
}
trap cleanup_resources EXIT INT TERM

search_service_is_ready() {
  python3 scripts/check_search_service.py \
    --url "$SEARCH_URL" \
    --timeout "${SEARCH_PREFLIGHT_TIMEOUT:-10}" \
    >/dev/null 2>&1
}

# Keep one retriever alive for the entire checkpoint sweep. Each nested
# eval_search_qa.sh invocation detects this service and therefore does not
# reload the E5 model and FAISS index.
if ! search_service_is_ready; then
  retriever_log="$output_root/search_retriever.log"
  bash examples/search/retriever/retrieval_launch.sh \
    >>"$retriever_log" 2>&1 &
  search_service_pid=$!

  startup_deadline=$((SECONDS + ${SEARCH_SERVICE_STARTUP_TIMEOUT:-1800}))
  while ! search_service_is_ready; do
    if ! kill -0 "$search_service_pid" 2>/dev/null; then
      wait "$search_service_pid" || true
      echo "SearchQA retrieval service exited during startup." >&2
      echo "Last ${SEARCH_RETRIEVER_ERROR_LINES:-80} log lines:" >&2
      tail -n "${SEARCH_RETRIEVER_ERROR_LINES:-80}" "$retriever_log" >&2 || true
      exit 1
    fi
    if (( SECONDS >= startup_deadline )); then
      echo "Timed out waiting for SearchQA retrieval service at $SEARCH_URL" >&2
      tail -n "${SEARCH_RETRIEVER_ERROR_LINES:-80}" "$retriever_log" >&2 || true
      exit 1
    fi
    sleep "${SEARCH_SERVICE_POLL_INTERVAL:-5}"
  done
fi

echo "SearchQA retrieval service is ready; starting two-way checkpoint sweep."
echo "Evaluation slot 0: GPUs $eval_gpu_group_0"
echo "Evaluation slot 1: GPUs $eval_gpu_group_1"
echo "Validation batch size: $eval_val_batch_size per slot, $((eval_val_batch_size * 2)) aggregate"

mapfile -t checkpoint_dirs < <(
  find "$run_dir" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -print \
    | sort -Vr
)

selected=0
completed=0
skipped=0
evaluation_dirs=()

for checkpoint_dir in "${checkpoint_dirs[@]}"; do
  checkpoint_name="${checkpoint_dir##*/}"
  step="${checkpoint_name#global_step_}"
  if ! [[ "$step" =~ ^[0-9]+$ ]]; then
    continue
  fi
  if (( step < start_step || step > end_step )); then
    continue
  fi

  selected=$((selected + 1))
  if [[ ! -d "$checkpoint_dir/actor" ]]; then
    echo "Skipping $checkpoint_name: actor directory not found." >&2
    skipped=$((skipped + 1))
    continue
  fi

  result_json="$output_root/$checkpoint_name.json"
  if [[ -s "$result_json" && "$force_reeval" != "1" ]]; then
    echo "Skipping $checkpoint_name: result already exists at $result_json"
    skipped=$((skipped + 1))
    continue
  fi

  evaluation_dirs+=("$checkpoint_dir")
done

if (( selected == 0 )); then
  echo "No global_step checkpoints found in range [$start_step, $end_step] under $run_dir" >&2
  exit 1
fi

evaluate_checkpoint() {
  local checkpoint_dir="$1"
  local gpu_group="$2"
  local slot="$3"
  local checkpoint_name="${checkpoint_dir##*/}"
  local checkpoint_output="$output_root/$checkpoint_name"
  local result_json="$output_root/$checkpoint_name.json"
  local summary_json="$checkpoint_output/val_traj/accuracy_summary.json"
  local temporary_json="$result_json.tmp"
  local ray_temp_dir
  local evaluation_status

  mkdir -p "$checkpoint_output"
  # Ray appends long session/socket names below _temp_dir. Keep this base path
  # deliberately short so the Linux AF_UNIX 107-byte socket limit is not hit.
  ray_temp_dir="$(mktemp -d "/tmp/searchqa_ray_${slot}_XXXXXX")"

  echo "[$checkpoint_name] Starting on GPUs $gpu_group (slot $slot)"
  set +e
  AUTO_TEST_AFTER_TRAIN=0 \
  OUTPUT_DIR="$checkpoint_output" \
  EVAL_CHECKPOINT="$checkpoint_dir" \
  TRAIN_CUDA_VISIBLE_DEVICES="$gpu_group" \
  N_GPUS=4 \
  VAL_BATCH_SIZE="$eval_val_batch_size" \
    bash scripts/eval_search_qa.sh "$engine" \
      +ray_init.address=local \
      "+ray_init._temp_dir=$ray_temp_dir" \
      +ray_init.include_dashboard=False
  evaluation_status=$?
  set -e

  case "$ray_temp_dir" in
    /tmp/searchqa_ray_*) rm -rf -- "$ray_temp_dir" ;;
    *) echo "[$checkpoint_name] Refusing to remove unexpected Ray path: $ray_temp_dir" >&2 ;;
  esac
  if (( evaluation_status != 0 )); then
    return "$evaluation_status"
  fi

  if [[ ! -s "$summary_json" ]]; then
    echo "[$checkpoint_name] Accuracy summary is missing: $summary_json" >&2
    return 1
  fi

  cp "$summary_json" "$temporary_json"
  mv "$temporary_json" "$result_json"
  echo "[$checkpoint_name] Saved accuracy: $result_json"
}

gpu_groups=("$eval_gpu_group_0" "$eval_gpu_group_1")
for ((batch_start = 0; batch_start < ${#evaluation_dirs[@]}; batch_start += 2)); do
  evaluation_pids=()
  evaluation_names=()

  for slot in 0 1; do
    checkpoint_index=$((batch_start + slot))
    if (( checkpoint_index >= ${#evaluation_dirs[@]} )); then
      break
    fi

    checkpoint_dir="${evaluation_dirs[$checkpoint_index]}"
    checkpoint_name="${checkpoint_dir##*/}"
    evaluate_checkpoint "$checkpoint_dir" "${gpu_groups[$slot]}" "$slot" &
    evaluation_pids+=("$!")
    evaluation_names+=("$checkpoint_name")
  done

  pair_failed=0
  for job_index in "${!evaluation_pids[@]}"; do
    pid="${evaluation_pids[$job_index]}"
    checkpoint_name="${evaluation_names[$job_index]}"
    if wait "$pid"; then
      completed=$((completed + 1))
    else
      echo "Evaluation failed for $checkpoint_name (pid $pid)." >&2
      pair_failed=1
    fi
  done
  evaluation_pids=()

  if (( pair_failed != 0 )); then
    echo "Stopping checkpoint sweep because at least one parallel evaluation failed." >&2
    exit 1
  fi
done

echo "Checkpoint evaluation complete: selected=$selected, completed=$completed, skipped=$skipped"
echo "Accuracy JSON files: $output_root/global_step_*.json"
