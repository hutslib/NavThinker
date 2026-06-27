#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
export PYTHONPATH=$(pwd)/../../..:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HYDRA_FULL_ERROR=1 

python -u -m habitat-baselines.habitat_baselines.run \
--config-name=social_nav_v2/orca_hm3d.yaml \
habitat_baselines.evaluate=True \
> evaluation/orca/hm3d/eval.log 2>&1