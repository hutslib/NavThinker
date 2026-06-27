#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import sys
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch

from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

## for import functions related to navthinker
import navthinker
from omegaconf import OmegaConf
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="evalai", path="input/")
        search_path.append(provider="evalai", path="team1/")
        search_path.append(provider="evalai", path="team2/")
        search_path.append(provider="evalai", path="team3/")
        search_path.append(provider="evalai", path="team4/")
        search_path.append(provider="evalai", path="team5/")

register_hydra_plugin(HabitatConfigPlugin)
@hydra.main(
    version_base=None,
    config_path="config",
    config_name="pointnav/ppo_pointnav_example",
)

def main(cfg: "DictConfig"):
    cfg = patch_config(cfg)

    # ========== ① Force configuration values ==========
    if cfg.habitat_baselines.evaluate != True:
        raise ValueError(
            f"[ERROR] habitat_baselines.evaluate must be True, but got {cfg.habitat_baselines.evaluate}"
        )

    if cfg.habitat.seed != 100:
        raise ValueError(
            f"[ERROR] habitat.seed must be 100, but got {cfg.habitat.seed}"
        )

    # if len(cfg.habitat_baselines.eval.video_option) != 0:
    #     raise ValueError(
    #         f"[ERROR] habitat_baselines.eval.video_option must be [''], but got {cfg.habitat_baselines.eval.video_option}"
    #     )

    # ========== ② Strict validation of observation keys ==========
    allowed_obs_keys = [
        "agent_0_articulated_agent_jaw_rgb",
        "agent_0_articulated_agent_jaw_depth",
        "agent_0_pointgoal_with_gps_compass",
        "agent_0_localization_sensor",
        "agent_1_localization_sensor",
        "agent_2_localization_sensor",
        "agent_3_localization_sensor",
        "agent_4_localization_sensor",
        "agent_5_localization_sensor",
        "agent_6_localization_sensor",
        "agent_0_oracle_shortest_path_sensor",
        "agent_0_human_velocity_sensor",
    ]
    obs_keys = cfg.habitat.gym.obs_keys
    invalid_obs_keys = set(obs_keys) - set(allowed_obs_keys)
    if invalid_obs_keys:
        raise ValueError(
            f"[obs_keys ERROR] Invalid obs_keys detected: {invalid_obs_keys}. "
            f"Only allowed keys are: {allowed_obs_keys}."
        )

    # ========== ③ Force task type ==========
    must_be_task_type = "MultiAgentPointNavTask-v0"
    if cfg.habitat.task.type != must_be_task_type:
        raise ValueError(
            f"[task.type ERROR] habitat.task.type must be '{must_be_task_type}', "
            f"but got '{cfg.habitat.task.type}'."
        )

    # ========== ④ Strict validation of measurements keys ==========
    allowed_measurements = [
        "distance_to_goal",
        "distance_to_goal_reward",
        "success",
        "did_multi_agents_collide",
        "num_steps",
        "top_down_map",
        "spl",
        "psc",
        "human_collision",
        "human_velocity_measure",
    ]
    measurement_keys = list(cfg.habitat.task.measurements.keys())
    invalid_measurements = set(measurement_keys) - set(allowed_measurements)
    if invalid_measurements:
        raise ValueError(
            f"[measurements ERROR] Invalid measurements detected: {invalid_measurements}. "
            f"Only allowed measurements are: {allowed_measurements}."
        )

    # ========== ⑤ Environment count constraint ==========
    # The number of environments can be set by the user, but must not exceed 8.
    # Single-environment execution (e.g., 1) is allowed for debugging or lightweight runs.
    # This provides flexibility while ensuring resource usage stays within limits.
    if cfg.habitat_baselines.num_environments > 8:
        raise ValueError(
            f"[ERROR] habitat_baselines.num_environments must be <= 8, but got {cfg.habitat_baselines.num_environments}"
        )

    execute_exp(cfg, "eval")

def execute_exp(config: "DictConfig", run_type: str) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)
    if (
        config.habitat_baselines.force_torch_single_threaded
        and torch.cuda.is_available()
    ):
        torch.set_num_threads(1)

    from habitat_baselines.common.baseline_registry import baseline_registry

    trainer_init = baseline_registry.get_trainer(
        config.habitat_baselines.trainer_name
    )
    assert (
        trainer_init is not None
    ), f"{config.habitat_baselines.trainer_name} is not supported"
    trainer = trainer_init(config)

    trainer.eval()


if __name__ == "__main__":
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    if "--exp-config" in sys.argv or "--run-type" in sys.argv:
        raise ValueError(
            "The API of run.py has changed to be compatible with hydra.\n"
            "--exp-config is now --config-name and is a config path inside habitat-baselines/habitat_baselines/config/. \n"
            "--run-type train is replaced with habitat_baselines.evaluate=False (default) and --run-type eval is replaced with habitat_baselines.evaluate=True.\n"
            "instead of calling:\n\n"
            "python -u -m habitat_baselines.run --exp-config habitat-baselines/habitat_baselines/config/<path-to-config> --run-type train/eval\n\n"
            "You now need to do:\n\n"
            "python -u -m habitat_baselines.run --config-name=<path-to-config> habitat_baselines.evaluate=False/True\n"
        )
    main()
