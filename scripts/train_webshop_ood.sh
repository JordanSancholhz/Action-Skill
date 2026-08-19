set -x
ENGINE=${1:-vllm}
shift  # Remove first argument so $@ only contains extra params
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=4,5,6,7

# ==================== WandB (optional) ====================
# export WANDB_API_KEY="your_key_here"

export VERL_DETERMINISTIC_SAMPLING=1
export VERL_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1

export RAY_BACKEND_LOG_LEVEL=warning
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DISABLE_GPU_MONITOR=1
export RAY_DEBUG_POST_MORTEM=0
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export RAY_ROTATION_MAX_BYTES=52428800
export RAY_ROTATION_BACKUP_COUNT=3
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"

# Limit thread count to prevent pthread_create failures under Ray
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export RAY_IGNORE_UNHANDLED_ERRORS=1
export RAY_worker_register_timeout_seconds=600
export RAY_TASK_MAX_RETRIES=3
export RAY_memory=100000000000
export RAY_object_store_memory=40000000000

# Ensure sentence-transformers and faiss-cpu are pre-installed
# pip3 install sentence-transformers faiss-cpu
python3 -m spacy download en_core_web_sm 2>/dev/null || true

# ==================== Model & Data Config ====================
export MODEL_PATH="${MODEL_PATH:?Please set MODEL_PATH to your Qwen2.5-7B-Instruct model path}"
export WEBSHOP_DATA="${WEBSHOP_DATA:?Please set WEBSHOP_DATA to the WebShop data directory}"

project_name=skill_webshop_ood
RUN_ID="$(date +%m%d-%H%M%S)"
experiment_name="skill05_webshop_ood${RUN_ID:+_$RUN_ID}"

export OUTPUT_DIR="./outputs/${project_name}/${experiment_name}"
mkdir -p "${OUTPUT_DIR}/logs"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
mkdir -p "$WANDB_DIR"

# ==================== Training Hyperparameters ====================
num_cpus_per_env_worker=0.1
train_data_size=16
val_data_size=1500
group_size=8

# Placeholder parquet already in $WEBSHOP_DATA/text/

python3 -m verl.trainer.main_ppo_webshop_ood \
    algorithm.adv_estimator=grpo \
    data.train_files=${WEBSHOP_DATA}/text/train.parquet \
    data.val_files=${WEBSHOP_DATA}/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=6000 \
    data.max_response_length=768 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    +actor_rollout_ref.rollout.sleep_at_rollout_end=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    env.env_name=Webshop \
    env.seed=0 \
    env.webshop.use_small=False \
    env.webshop.human_goals=True \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=memory_data/webshop_ood/claude_style_skills_id.json \
    +env.skills_only_memory.retrieval_mode=embedding \
    +env.skills_only_memory.embedding_model_path=models/Qwen3-Embedding-0.6B \
    +env.skills_only_memory.top_k=3 \
    +env.skills_only_memory.enable_dynamic_update=False \
    +env.webshop_ood.id_categories='[apparel,footwear,electronics,other]' \
    +env.webshop_ood.ood_categories='[home_decor,accessories,beauty_health]' \
    +env.webshop_ood.skills_json_path=memory_data/webshop_ood/claude_style_skills_ood.json \
    +env.webshop_ood.splits_file=${WEBSHOP_DATA}/webshop_ood_splits.json \
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
    +env.ours.dpo_top_k=2 \
    +env.ours.dpo_min_score_gap=0.1 \
    +env.ours.continuous_score_key=trajectory_webshop_task_score \
    +env.ours.filtered_negative_advantage_weight=1.0 \
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
    +env.internalize.jsd_micro_batch_size_per_gpu=2 \
    +env.internalize.action_ig_beta=0.2 \
    +env.internalize.action_ig_clip=1.2 \
    +env.guide_internalize=True \
    actor_rollout_ref.actor.ppo_epochs=1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.rollout_data_dir=${OUTPUT_DIR}/rollout_data \
    +trainer.val_dump_path=${OUTPUT_DIR}/val_traj \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=5 \
    trainer.total_epochs=999 \
    trainer.total_training_steps=150 \
    trainer.val_before_train=False $@ 2>&1 | tee "${OUTPUT_DIR}/logs/$(date +%Y%m%d_%H%M%S).log"
