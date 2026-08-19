#!/usr/bin/env bash
set -euo pipefail

ENGINE="${1:-vllm}"
if [[ $# -gt 0 ]]; then
  shift
fi

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VERL_DETERMINISTIC_SAMPLING="${VERL_DETERMINISTIC_SAMPLING:-1}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-WARNING}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export RAY_BACKEND_LOG_LEVEL="${RAY_BACKEND_LOG_LEVEL:-warning}"
export RAY_DISABLE_IMPORT_WARNING="${RAY_DISABLE_IMPORT_WARNING:-1}"
export RAY_IGNORE_UNHANDLED_ERRORS="${RAY_IGNORE_UNHANDLED_ERRORS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export HF_DATASETS_DISABLE_PROGRESS_BARS="${HF_DATASETS_DISABLE_PROGRESS_BARS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
# Keep the retrieval service on the first four physical GPUs and reserve the
# last four physical GPUs for Ray/vLLM training.
export SEARCH_CUDA_VISIBLE_DEVICES=0,1,2,3
export TRAIN_CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES"
export N_GPUS="${N_GPUS:-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-512}"
export GROUP_SIZE="${GROUP_SIZE:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-512}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-16}"
export JSD_MICRO_BATCH_SIZE_PER_GPU="${JSD_MICRO_BATCH_SIZE_PER_GPU:-1}"
# Action-information scoring materializes full-vocabulary logits. With Qwen2.5
# and long SearchQA trajectories, 64 requested about 83.5 GiB at once; 1 keeps
# the same computation while leaving ample room for both logits and JSD buffers.
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
export AUTO_TEST_AFTER_TRAIN="${AUTO_TEST_AFTER_TRAIN:-1}"
export AUTO_PERIODIC_TEST_AFTER_TRAIN="${AUTO_PERIODIC_TEST_AFTER_TRAIN:-1}"
export PERIODIC_TEST_START_STEP="${PERIODIC_TEST_START_STEP:-90}"
export PERIODIC_TEST_FREQ="${PERIODIC_TEST_FREQ:-15}"
export SEARCH_PREFLIGHT="${SEARCH_PREFLIGHT:-1}"
export AUTO_START_SEARCH_SERVICE="${AUTO_START_SEARCH_SERVICE:-1}"
export SEARCH_SERVICE_STARTUP_TIMEOUT="${SEARCH_SERVICE_STARTUP_TIMEOUT:-1800}"
export SEARCH_SERVICE_POLL_INTERVAL="${SEARCH_SERVICE_POLL_INTERVAL:-5}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-True}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4608}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"

python3 - "$VLLM_GPU_MEMORY_UTILIZATION" <<'PY'
import sys

try:
    value = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit(f"VLLM_GPU_MEMORY_UTILIZATION must be numeric: {sys.argv[1]!r}") from exc
if not 0.0 < value <= 0.5:
    raise SystemExit(
        "VLLM_GPU_MEMORY_UTILIZATION must be greater than 0 and no greater than 0.5; "
        f"got {value}"
    )
PY

MODEL_PATH="${MODEL_PATH:?Please set MODEL_PATH to the base instruct model}"
SEARCH_DATA="${SEARCH_DATA:-$PWD/search_data}"
SEARCH_QA_DATA="${SEARCH_QA_DATA:-$SEARCH_DATA/text}"
if [[ -z "${SEARCH_QA_VAL_FILE:-}" ]]; then
  if [[ -f "$SEARCH_QA_DATA/val.parquet" ]]; then
    SEARCH_QA_VAL_FILE="$SEARCH_QA_DATA/val.parquet"
  else
    SEARCH_QA_VAL_FILE="$SEARCH_QA_DATA/test.parquet"
  fi
fi
SEARCH_PORT="${SEARCH_PORT:-8030}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:${SEARCH_PORT}/retrieve}"
SEARCH_RETRIEVER_PATH="${SEARCH_RETRIEVER_PATH:-models/e5-base-v2}"
SEARCH_SKILLS_FILE="${SEARCH_SKILLS_FILE:-memory_data/search_qa/claude_style_skills.json}"
SEARCH_SKILL_MODE="${SEARCH_SKILL_MODE:-template}"
SEARCH_EMBEDDING_MODEL="${SEARCH_EMBEDDING_MODEL:-models/Qwen3-Embedding-0.6B}"

required_files=(
  "$SEARCH_QA_DATA/train.parquet"
  "$SEARCH_QA_VAL_FILE"
  "$SEARCH_SKILLS_FILE"
)
if [[ "$AUTO_TEST_AFTER_TRAIN" == "1" || "$AUTO_PERIODIC_TEST_AFTER_TRAIN" == "1" ]]; then
  required_files+=("$SEARCH_QA_DATA/test.parquet")
fi
for required_file in "${required_files[@]}"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required file not found: $required_file" >&2
    exit 1
  fi
done

project_name="${PROJECT_NAME:-skill_search_qa}"
run_id="$(date +%m%d-%H%M%S)"
experiment_name="${EXPERIMENT_NAME:-skill05_search_qa_$run_id}"
output_dir="${OUTPUT_DIR:-./outputs/$project_name/$experiment_name}"
mkdir -p "$output_dir/logs" "$output_dir/wandb"
export WANDB_DIR="$output_dir/wandb"

search_service_pid=""
cleanup_search_service() {
  if [[ -n "$search_service_pid" ]] && kill -0 "$search_service_pid" 2>/dev/null; then
    kill "$search_service_pid" 2>/dev/null || true
    wait "$search_service_pid" 2>/dev/null || true
  fi
}
trap cleanup_search_service EXIT INT TERM

search_service_is_ready() {
  python3 scripts/check_search_service.py \
    --url "$SEARCH_URL" \
    --timeout "${SEARCH_PREFLIGHT_TIMEOUT:-10}" \
    >/dev/null 2>&1
}

if [[ "$SEARCH_PREFLIGHT" == "1" ]]; then
  if search_service_is_ready; then
    :
  elif [[ "$AUTO_START_SEARCH_SERVICE" == "1" ]]; then
    retriever_log="$output_dir/logs/search_retriever.log"
    SEARCH_DATA="$SEARCH_DATA" \
    SEARCH_PORT="$SEARCH_PORT" \
    SEARCH_RETRIEVER_PATH="$SEARCH_RETRIEVER_PATH" \
    CUDA_VISIBLE_DEVICES="$SEARCH_CUDA_VISIBLE_DEVICES" \
      bash examples/search/retriever/retrieval_launch.sh \
      >>"$retriever_log" 2>&1 &
    search_service_pid=$!

    search_service_deadline=$((SECONDS + SEARCH_SERVICE_STARTUP_TIMEOUT))
    while ! search_service_is_ready; do
      if ! kill -0 "$search_service_pid" 2>/dev/null; then
        wait "$search_service_pid" || true
        echo "SearchQA retrieval service exited before becoming ready." >&2
        echo "See retriever log: $retriever_log" >&2
        echo "Last ${SEARCH_RETRIEVER_ERROR_LINES:-80} lines:" >&2
        tail -n "${SEARCH_RETRIEVER_ERROR_LINES:-80}" "$retriever_log" >&2 || true
        exit 1
      fi
      if (( SECONDS >= search_service_deadline )); then
        echo "Timed out after ${SEARCH_SERVICE_STARTUP_TIMEOUT}s waiting for $SEARCH_URL" >&2
        echo "See retriever log: $retriever_log" >&2
        echo "Last ${SEARCH_RETRIEVER_ERROR_LINES:-80} lines:" >&2
        tail -n "${SEARCH_RETRIEVER_ERROR_LINES:-80}" "$retriever_log" >&2 || true
        exit 1
      fi
      sleep "$SEARCH_SERVICE_POLL_INTERVAL"
    done
  else
    python3 scripts/check_search_service.py \
      --url "$SEARCH_URL" \
      --timeout "${SEARCH_PREFLIGHT_TIMEOUT:-10}"
  fi
fi

train_batch_size="$TRAIN_BATCH_SIZE"
val_batch_size="$VAL_BATCH_SIZE"
group_size="$GROUP_SIZE"
total_training_steps="${TOTAL_TRAINING_STEPS:-150}"
n_gpus="$N_GPUS"

skill_args=(
  "+env.use_skills_only_memory=True"
  "+env.skills_only_memory.skills_json_path=$SEARCH_SKILLS_FILE"
  "+env.skills_only_memory.retrieval_mode=$SEARCH_SKILL_MODE"
  "+env.skills_only_memory.top_k=3"
  "+env.skills_only_memory.task_specific_top_k=4"
  "+env.skills_only_memory.enable_dynamic_update=False"
)
if [[ "$SEARCH_SKILL_MODE" == "embedding" ]]; then
  skill_args+=("+env.skills_only_memory.embedding_model_path=$SEARCH_EMBEDDING_MODEL")
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$SEARCH_QA_DATA/train.parquet" \
  data.val_files="$SEARCH_QA_VAL_FILE" \
  data.train_batch_size="$train_batch_size" \
  data.val_batch_size="$val_batch_size" \
  data.max_prompt_length=4096 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=False \
  data.truncation=right \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name="$ENGINE" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  +actor_rollout_ref.rollout.sleep_at_rollout_end=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
  algorithm.use_kl_in_reward=False \
  env.env_name=search \
  env.seed=0 \
  env.max_steps=4 \
  env.history_length=4 \
  env.rollout.n="$group_size" \
  env.resources_per_worker.num_cpus=0.1 \
  env.search.search_url="$SEARCH_URL" \
  env.search.topk=3 \
  env.search.timeout=60 \
  env.search.log_requests=False \
  env.search.search_reward_coef=0.0 \
  +env.guide_internalize=True \
  "${skill_args[@]}" \
  +env.ours.warmup_steps=0 \
  +env.ours.action_ig_ramp_steps=5 \
  +env.ours.curriculum_enabled=True \
  +env.ours.curriculum_internalize_fraction=0.2 \
  +env.ours.curriculum_utilize_start_fraction=0.7 \
  +env.ours.curriculum_internalize_grpo_floor=1.0 \
  +env.ours.curriculum_internalize_max_pass_rate=0.0 \
  +env.ours.curriculum_min_aux_weight=0.01 \
  +env.ours.curriculum_jsd_clip_ratio_guard=0.05 \
  +env.ours.curriculum_jsd_mean_length_ratio_guard=0.4 \
  +env.ours.window_size=5 \
  +env.ours.std_window_init='[]' \
  +env.ours.success_threshold_default=0.5 \
  +env.ours.std_threshold_default=0.25 \
  +env.ours.dpo_beta=0.1 \
  +env.ours.hard_grpo_enabled=True \
  +env.ours.action_ig_enabled=True \
  +env.ours.success_threshold=0.0 \
  +env.utilize.omega=1.0 \
  +env.utilize.delta_window_size=5 \
  +env.utilize.adv2_clip=3.0 \
  +env.utilize.action_info_lambda=0.1 \
  +env.utilize.action_info_max_weight_delta=0.2 \
  +env.utilize.action_info_z_clip=3.0 \
  +env.utilize.action_info_gamma=1.0 \
  +env.utilize.action_info_adv_clip=3.0 \
  +env.utilize.action_info_fixed_scale=0.2 \
  +env.utilize.action_info_top_k=64 \
  +env.utilize.action_info_temperature=1.0 \
  +env.utilize.trajectory_balanced_loss=False \
  +env.internalize.jsd_lambda=1.0 \
  +env.internalize.action_jsd_lambda=0.1 \
  +env.internalize.jsd_top_k=64 \
  +env.internalize.jsd_temperature=1.0 \
  +env.internalize.jsd_micro_batch_size_per_gpu="$JSD_MICRO_BATCH_SIZE_PER_GPU" \
  +env.internalize.action_ig_beta=0.2 \
  +env.internalize.action_ig_clip=1.2 \
  +env.guide_internalize=True \
  actor_rollout_ref.actor.ppo_epochs=1 \
  trainer.critic_warmup=0 \
  trainer.logger="['console']" \
  +trainer.quiet_training=False \
  +trainer.print_step_time=True \
  +trainer.print_validation_metrics=False \
  trainer.project_name="$project_name" \
  trainer.experiment_name="$experiment_name" \
  trainer.default_local_dir="$output_dir" \
  trainer.rollout_data_dir="$output_dir/rollout_data" \
  +trainer.val_dump_path="$output_dir/val_traj" \
  trainer.n_gpus_per_node="$n_gpus" \
  trainer.nnodes=1 \
  trainer.save_freq=15 \
  trainer.test_freq=5 \
  trainer.select_best_checkpoint=True \
  trainer.best_checkpoint_metric=val/search_qa/accuracy \
  trainer.best_checkpoint_mode=max \
  trainer.total_epochs=999 \
  trainer.total_training_steps="$total_training_steps" \
  trainer.val_before_train=False \
  "$@" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.enforce_eager="$VLLM_ENFORCE_EAGER" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$VLLM_MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.max_num_seqs="$VLLM_MAX_NUM_SEQS" \
  2>&1 | tee "$output_dir/logs/$(date +%Y%m%d_%H%M%S).log"

if [[ "$AUTO_PERIODIC_TEST_AFTER_TRAIN" == "1" ]]; then
  if ! [[ "$PERIODIC_TEST_START_STEP" =~ ^[0-9]+$ ]] || \
     ! [[ "$PERIODIC_TEST_FREQ" =~ ^[1-9][0-9]*$ ]]; then
    echo "PERIODIC_TEST_START_STEP must be non-negative and PERIODIC_TEST_FREQ must be positive." >&2
    exit 1
  fi

  for ((test_step=PERIODIC_TEST_START_STEP; test_step<=total_training_steps; test_step+=PERIODIC_TEST_FREQ)); do
    checkpoint="$output_dir/global_step_$test_step"
    if [[ ! -d "$checkpoint/actor" ]]; then
      echo "Skipping periodic test: actor checkpoint not found at $checkpoint/actor" >&2
      continue
    fi

    test_output_dir="$output_dir/test_step_$test_step"
    echo "Evaluating SearchQA test set at training step $test_step: $checkpoint"
    AUTO_TEST_AFTER_TRAIN=0 \
    AUTO_PERIODIC_TEST_AFTER_TRAIN=0 \
    OUTPUT_DIR="$test_output_dir" \
    EVAL_CHECKPOINT="$checkpoint" \
      bash scripts/eval_search_qa.sh "$ENGINE"

    metrics_file="$test_output_dir/val_traj/metrics.json"
    if [[ -f "$metrics_file" ]]; then
      cp "$metrics_file" "$output_dir/test_metrics_step_$test_step.json"
    fi
    accuracy_file="$test_output_dir/val_traj/accuracy_summary.json"
    if [[ -f "$accuracy_file" ]]; then
      cp "$accuracy_file" "$output_dir/test_accuracy_step_$test_step.json"
    fi
  done
fi

if [[ "$AUTO_TEST_AFTER_TRAIN" == "1" ]]; then
  best_checkpoint_file="$output_dir/best_checkpoint.txt"
  if [[ ! -f "$best_checkpoint_file" ]]; then
    echo "Best-checkpoint marker not found: $best_checkpoint_file" >&2
    exit 1
  fi
  best_checkpoint="$(tr -d '\r\n' < "$best_checkpoint_file")"
  if [[ ! -d "$best_checkpoint/actor" ]]; then
    echo "Best actor checkpoint not found: $best_checkpoint/actor" >&2
    exit 1
  fi

  test_output_dir="$output_dir/test_best_checkpoint"
  echo "Evaluating full SearchQA test set with best checkpoint: $best_checkpoint"
  AUTO_TEST_AFTER_TRAIN=0 \
  AUTO_PERIODIC_TEST_AFTER_TRAIN=0 \
  OUTPUT_DIR="$test_output_dir" \
  EVAL_CHECKPOINT="$best_checkpoint" \
    bash scripts/eval_search_qa.sh "$ENGINE"

  metrics_file="$test_output_dir/val_traj/metrics.json"
  if [[ -f "$metrics_file" ]]; then
    cp "$metrics_file" "$output_dir/test_metrics_best_checkpoint.json"
    echo "Best-checkpoint test metrics: $output_dir/test_metrics_best_checkpoint.json"
  fi
  accuracy_file="$test_output_dir/val_traj/accuracy_summary.json"
  if [[ -f "$accuracy_file" ]]; then
    cp "$accuracy_file" "$output_dir/test_accuracy_best_checkpoint.json"
    echo "Best-checkpoint accuracy summary: $output_dir/test_accuracy_best_checkpoint.json"
  fi
fi
