#!/bin/bash
# Evaluate a trained NavThinker checkpoint on the val split.
#
# Usage:
#   bash scripts/eval.sh [config_name] <checkpoint.pth> [num_envs]
#   bash scripts/eval.sh navthinker_hm3d_eval.yaml experiments/navthinker/checkpoints/ckpt.100.pth 1
#
# Videos are written under eval_experiments/<config>/video/.
set -e

CONFIG_NAME="${1:-navthinker_hm3d_eval.yaml}"
CKPT_PATH="${2:?Usage: bash scripts/eval.sh <config_name> <checkpoint.pth> [num_envs]}"
NUM_ENVS="${3:-1}"

export PYTHONPATH="$(pwd):$(pwd)/habitat-baselines:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
export GLOG_minloglevel="${GLOG_minloglevel:-2}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

echo "Evaluating ${CONFIG_NAME} with checkpoint ${CKPT_PATH} (${NUM_ENVS} env)"

python -u -m habitat_baselines.run \
    --config-name="$CONFIG_NAME" \
    habitat_baselines.evaluate=True \
    habitat_baselines.load_resume_state_config=False \
    habitat_baselines.eval_ckpt_path_dir="$CKPT_PATH" \
    habitat_baselines.eval.should_load_ckpt=True \
    habitat_baselines.num_environments="$NUM_ENVS" \
    'habitat_baselines.eval.video_option=[disk]'
