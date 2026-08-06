#!/usr/bin/env bash
set -euo pipefail

SEARCH_DATA="${SEARCH_DATA:-$PWD/search_data}"
SEARCH_QA_DATA="${SEARCH_QA_DATA:-$SEARCH_DATA/text}"
SEARCH_QA_REPO="${SEARCH_QA_REPO:-PeterJinGo/nq_hotpotqa_train}"
SEARCH_QA_RAW_DIR="${SEARCH_QA_RAW_DIR:-}"
VAL_SAMPLES_PER_GROUP="${VAL_SAMPLES_PER_GROUP:-1000}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-}"

sample_args=()
if [[ -n "$SEARCH_QA_RAW_DIR" ]]; then
  sample_args+=(--source_dir "$SEARCH_QA_RAW_DIR")
fi
if [[ -n "$MAX_TRAIN_SAMPLES" ]]; then
  sample_args+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
fi
if [[ -n "$MAX_TEST_SAMPLES" ]]; then
  sample_args+=(--max_test_samples "$MAX_TEST_SAMPLES")
fi

python3 examples/data_preprocess/preprocess_search_r1_dataset.py \
  --hf_repo_id "$SEARCH_QA_REPO" \
  --local_dir "$SEARCH_QA_DATA" \
  "${sample_args[@]}"

python3 examples/data_preprocess/split_search_qa_id_ood.py \
  --input "$SEARCH_QA_DATA/test.parquet" \
  --output_dir "$SEARCH_QA_DATA" \
  --val_samples_per_group "$VAL_SAMPLES_PER_GROUP"

echo "Search-QA data is ready under: $SEARCH_QA_DATA"
