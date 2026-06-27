#!/bin/bash
# Train the NavThinker policy (multi-GPU DD-PPO).
#
# Usage:
#   bash scripts/train.sh [config_name] [num_gpus]
#   bash scripts/train.sh navthinker_hm3d.yaml 4
#
# Configs are resolved from the repo-root `configs/` directory.
# Re-running resumes from the latest checkpoint under experiments/<config>/.
set -e

CONFIG_NAME="${1:-navthinker_hm3d.yaml}"
TOTAL_GPU="${2:-${TOTAL_GPU:-4}}"
EXP_NAME="$(basename "$CONFIG_NAME" .yaml)"

export PYTHONPATH="$(pwd):$(pwd)/habitat-baselines:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
export GLOG_minloglevel="${GLOG_minloglevel:-2}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

mkdir -p "experiments/${EXP_NAME}"/{tb,video,checkpoints}

echo "Training ${EXP_NAME} on ${TOTAL_GPU} GPU(s) (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"

python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node "$TOTAL_GPU" \
    --master_port "$MASTER_PORT" \
    habitat-baselines/habitat_baselines/run.py \
    --config-name="$CONFIG_NAME" \
    habitat_baselines.load_resume_state_config=True
