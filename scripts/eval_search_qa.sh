#!/usr/bin/env bash
set -euo pipefail

# Full evaluation uses the combined test set; ray_trainer reports separate
# val/id/* and val/ood/* aggregates as well as per-source metrics.
SEARCH_DATA="${SEARCH_DATA:-$PWD/search_data}"
SEARCH_QA_DATA="${SEARCH_QA_DATA:-$SEARCH_DATA/text}"
export SEARCH_QA_VAL_FILE="${SEARCH_QA_VAL_FILE:-$SEARCH_QA_DATA/test.parquet}"

# Reuse the exact training configuration, but stop after the initial validation.
checkpoint_args=()
if [[ -n "${EVAL_CHECKPOINT:-}" ]]; then
  if [[ ! -d "$EVAL_CHECKPOINT/actor" ]]; then
    echo "Actor checkpoint not found: $EVAL_CHECKPOINT/actor" >&2
    exit 1
  fi
  checkpoint_args=(
    "trainer.resume_mode=resume_path"
    "trainer.resume_from_path=$EVAL_CHECKPOINT"
  )
fi

AUTO_TEST_AFTER_TRAIN=0 AUTO_PERIODIC_TEST_AFTER_TRAIN=0 \
  exec bash scripts/train_search_qa.sh "${1:-vllm}" \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.select_best_checkpoint=False \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=1 \
  "${checkpoint_args[@]}" \
  "${@:2}"
