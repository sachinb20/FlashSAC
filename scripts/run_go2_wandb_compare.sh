#!/bin/bash
##################################################################################
# Go2 walk: FlashSAC-vanilla (Genesis) vs. the ported IsaacLab env, both logging to
# Weights & Biases with video. Same group_name so both runs land side-by-side in the
# wandb UI -- their episode-reward-term keys are named identically (Reward/<term>,
# episode_length, avg_return, ...), so panels overlay directly.
#
# num_env_steps / num_train_envs / buffer sizes and the four logging cadences
# (evaluation_per_interaction_step, metrics_per_interaction_step,
# recording_per_interaction_step, logging_per_interaction_step) are all pinned to
# identical explicit values below rather than left to derive from the formulas in
# flashSAC_base.yaml, so the two runs are guaranteed to log metrics and video on the
# same schedule even if those base-config defaults change later.
#
# entity_name is left unset here so wandb.init() falls back to your account's default
# entity (see flash_rl/common/logger.py); pass --overrides entity_name=<your-team> if
# you want a specific one.
#
# NOTE: for isaaclab_go2, train/eval/record share one sim instance, so
# env.enable_cameras=true + env.record_video=true add render overhead to every training
# step, not just the recording passes (see configs/env/isaaclab_go2.yaml).
##################################################################################
set -euo pipefail

seed=${1:-0}
group_name=${2:-go2-walk-compare}

# shared across both runs -- edit once here, both commands stay in sync
num_env_steps=50_000_896
num_train_envs=1024
buffer_max_length=10_000_000
buffer_min_length=100_000
evaluation_per_interaction_step=4882
metrics_per_interaction_step=4882
recording_per_interaction_step=4882
logging_per_interaction_step=488

echo "=== Genesis (vanilla FlashSAC) go2-walk, seed ${seed} ==="
uv run python train.py \
    --config_name flashSAC_base \
    --overrides seed=${seed} \
    --overrides group_name=${group_name} \
    --overrides exp_name=genesis \
    `#=== Logging ===#` \
    --overrides logger_type=wandb \
    --overrides entity_name=null \
    --overrides evaluation_per_interaction_step=${evaluation_per_interaction_step} \
    --overrides metrics_per_interaction_step=${metrics_per_interaction_step} \
    --overrides recording_per_interaction_step=${recording_per_interaction_step} \
    --overrides logging_per_interaction_step=${logging_per_interaction_step} \
    `#=== Environment (GPU sim) ===#` \
    --overrides env=genesis \
    --overrides env.env_name=go2-walk \
    --overrides num_env_steps=${num_env_steps} \
    --overrides num_train_envs=${num_train_envs} \
    --overrides num_eval_envs=null \
    --overrides num_record_envs=null \
    --overrides num_eval_episodes=1024 \
    --overrides num_record_episodes=1 \
    `#=== Agent (GPU sim) ===#` \
    --overrides agent=flashSAC \
    --overrides agent.buffer_max_length=${buffer_max_length} \
    --overrides agent.buffer_min_length=${buffer_min_length} \
    --overrides agent.buffer_device_type='cuda' \
    --overrides agent.sample_batch_size=2048 \
    --overrides agent.use_amp=true \
    --overrides updates_per_interaction_step=2 \
    `#=== Benchmark default ===#` \
    --overrides agent.asymmetric_observation=true \
    --overrides gamma=0.95 \
    --overrides n_step=1

echo "=== IsaacLab Go2 (ported env) go2-vel-direct, seed ${seed} ==="
uv run python train.py \
    --config_name flashSAC_base \
    --overrides seed=${seed} \
    --overrides group_name=${group_name} \
    --overrides exp_name=isaaclab_go2 \
    `#=== Logging ===#` \
    --overrides logger_type=wandb \
    --overrides entity_name=null \
    --overrides evaluation_per_interaction_step=${evaluation_per_interaction_step} \
    --overrides metrics_per_interaction_step=${metrics_per_interaction_step} \
    --overrides recording_per_interaction_step=${recording_per_interaction_step} \
    --overrides logging_per_interaction_step=${logging_per_interaction_step} \
    `#=== Environment (GPU sim) ===#` \
    --overrides env=isaaclab_go2 \
    --overrides env.enable_cameras=true \
    --overrides env.record_video=true \
    --overrides env.isaac_video_camera_mode=single_env \
    --overrides env.isaac_direct_velocity_terrain_mode=flat \
    --overrides num_env_steps=${num_env_steps} \
    --overrides num_train_envs=${num_train_envs} \
    --overrides num_eval_envs=null \
    --overrides num_record_envs=null \
    --overrides num_eval_episodes=1024 \
    --overrides num_record_episodes=1 \
    `#=== Agent (GPU sim) ===#` \
    --overrides agent=flashSAC \
    --overrides agent.buffer_max_length=${buffer_max_length} \
    --overrides agent.buffer_min_length=${buffer_min_length} \
    --overrides agent.buffer_device_type='cuda' \
    --overrides agent.sample_batch_size=2048 \
    --overrides agent.use_amp=true \
    --overrides updates_per_interaction_step=2 \
    `#=== Benchmark default ===#` \
    --overrides agent.asymmetric_observation=true \
    --overrides gamma=0.95 \
    --overrides n_step=1
