set -e
set -x

# Resume from global_step_60 with the smooth unified curriculum.
#
# Usage:
#   bash scripts/train_alfworld_ood_60.sh vllm [checkpoint] [hydra overrides...]
#
# The checkpoint can also be supplied through RESUME_CHECKPOINT. If neither is
# supplied, the script uses the step-60 checkpoint shown in results/method3.log.

ENGINE=${1:-vllm}
if [ "$#" -gt 0 ]; then
    shift
fi
export CUDA_VISIBLE_DEVICES=4,5,6,7

DEFAULT_CHECKPOINT="./outputs/skill_alfworld_ood/skill05_alfworld_ood_0720-155658/global_step_60"
CHECKPOINT="${RESUME_CHECKPOINT:-}"
if [ -z "$CHECKPOINT" ] && [ "$#" -gt 0 ] && [[ "$1" != *=* ]]; then
    CHECKPOINT="$1"
    shift
fi
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CHECKPOINT}"

export EXPECTED_CHECKPOINT_STEP=60
export RESUME_CHECKPOINT="$CHECKPOINT"
export OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR_OVERRIDE:-./outputs/outputs1}"
export WANDB_NAME_OVERRIDE="${WANDB_NAME_OVERRIDE:-skill05_alfworld_ood_stage60_action_info}"
export SAVE_FREQ="${SAVE_FREQ:-30}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
export OURS_WARMUP_STEPS=0
export OURS_CURRICULUM_ENABLED="${OURS_CURRICULUM_ENABLED:-True}"
export OURS_HARD_GRPO_ENABLED="${OURS_HARD_GRPO_ENABLED:-True}"
export ACTION_INFO_FIXED_SCALE="${ACTION_INFO_FIXED_SCALE:-0.2}"
export ACTION_INFO_MAX_WEIGHT_DELTA="${ACTION_INFO_MAX_WEIGHT_DELTA:-0.2}"
export ACTION_INFO_Z_CLIP="${ACTION_INFO_Z_CLIP:-3.0}"
export CURRICULUM_INTERNALIZE_FRACTION="${CURRICULUM_INTERNALIZE_FRACTION:-0.2}"
export CURRICULUM_UTILIZE_START_FRACTION="${CURRICULUM_UTILIZE_START_FRACTION:-0.7}"
export CURRICULUM_INTERNALIZE_GRPO_FLOOR="${CURRICULUM_INTERNALIZE_GRPO_FLOOR:-1.0}"
export CURRICULUM_INTERNALIZE_MAX_PASS_RATE="${CURRICULUM_INTERNALIZE_MAX_PASS_RATE:-0.0}"
export CURRICULUM_MIN_AUX_WEIGHT="${CURRICULUM_MIN_AUX_WEIGHT:-0.01}"
export CURRICULUM_JSD_CLIP_RATIO_GUARD="${CURRICULUM_JSD_CLIP_RATIO_GUARD:-0.05}"
export CURRICULUM_JSD_MEAN_LENGTH_RATIO_GUARD="${CURRICULUM_JSD_MEAN_LENGTH_RATIO_GUARD:-0.4}"
export JSD_MICRO_BATCH_SIZE_PER_GPU="${JSD_MICRO_BATCH_SIZE_PER_GPU:-4}"
# Old checkpoints do not contain the new success/std history windows. Leave
# them empty by default and use the configured cold-start thresholds.
export ROUTING_WINDOW_INIT="${ROUTING_WINDOW_INIT:-[]}"
export STD_WINDOW_INIT="${STD_WINDOW_INIT:-[]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/train_alfworld_ood2.sh" "$ENGINE" "$@"
