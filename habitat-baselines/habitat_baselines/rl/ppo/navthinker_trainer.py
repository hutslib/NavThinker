#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import math
import contextlib
import os
import random
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import habitat_baselines.rl.multi_agent  # noqa: F401.
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.utils import profiling_wrapper
from habitat_baselines.common import VectorEnvFactory
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_writer,
)
from habitat_baselines.rl.ddppo.algo import DDPPO  # noqa: F401.
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    SAVE_STATE,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr
from habitat_baselines.rl.ppo.evaluator import Evaluator
from habitat_baselines.rl.ppo.single_agent_access_mgr import (  # noqa: F401.
    SingleAgentAccessMgr,
)
from habitat_baselines.utils.common import (
    batch_obs,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    NON_SCALAR_METRICS,
    extract_scalars_from_infos,
)
from habitat_baselines.utils.timing import g_timer
from habitat_baselines.utils.wm_visualizer import (
    render_depth_comparison,
    render_trajectory_comparison,
    render_feature_pca_comparison,
    render_probe_depth_comparison,
    render_depth_triple,
    render_cosine_sim_heatmap,
    render_history_panel,
    render_rgb_observation,
    render_step_label,
    LinearDepthProbe,
    compose_wm_frame,
    vstack_wm_panels,
)

def _rebuild_obs_space_from_meta(obs_space_meta: Dict) -> "gym.spaces.Dict":
    """Reconstruct a gym observation space from metadata saved with the replay buffer."""
    import gym.spaces as spaces

    space_dict = {}
    for key, meta in obs_space_meta.items():
        shape = tuple(meta["shape"])
        dtype_str = meta["dtype"]
        if "float" in dtype_str:
            space_dict[key] = spaces.Box(
                low=-np.inf, high=np.inf, shape=shape, dtype=np.float32,
            )
        elif "uint8" in dtype_str:
            space_dict[key] = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8,
            )
        elif "int" in dtype_str:
            space_dict[key] = spaces.Box(
                low=-np.iinfo(np.int64).max,
                high=np.iinfo(np.int64).max,
                shape=shape,
                dtype=np.int64,
            )
        else:
            space_dict[key] = spaces.Box(
                low=-np.inf, high=np.inf, shape=shape, dtype=np.float32,
            )
    return spaces.Dict(space_dict)


def contains_inf_or_nan(observations):
    for key, value in observations.items():
        if isinstance(value, (float, int)):
            # 如果是标量，检查是否为 NaN 或 inf
            if math.isinf(value) or math.isnan(value):
                print(f"Key {key} contains inf or nan: {value}")
                return True
        elif isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
            # 如果是列表、数组或张量，检查每个元素是否为 NaN 或 inf
            if isinstance(value, torch.Tensor):
                if torch.isinf(value).any() or torch.isnan(value).any():
                    print(f"Key {key} contains inf or nan in tensor")
                    return True
            elif isinstance(value, np.ndarray):
                if np.isinf(value).any() or np.isnan(value).any():
                    print(f"Key {key} contains inf or nan in numpy array")
                    return True
            else:
                for element in value:
                    if isinstance(element, (float, int)) and (math.isinf(element) or math.isnan(element)):
                        print(f"Key {key} contains inf or nan in list/tuple: {element}")
                        return True
    return False

class _WorldModelTrainModule(torch.nn.Module):
    """DDP-friendly wrapper that computes WM losses in one forward."""

    def __init__(self, world_model: torch.nn.Module):
        super().__init__()
        self.world_model = world_model

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        kl_free_bits: float,
        kl_dyn_scale: float = 0.5,
        kl_rep_scale: float = 0.1,
        imag_loss_scale: float = 0.0,
        imag_loss_depth_scale: float = 1.0,
        imag_loss_reward_scale: float = 1.0,
        imag_loss_traj_scale: float = 1.0,
        imag_loss_horizon_gamma: float = 0.95,
        imag_loss_horizon: int = 5,
        feature_loss_scale: float = 0.0,
        depth_pred_loss_scale: float = 0.0,
    ):
        # Encode observations: (B, T, ...) -> (B*T, ...)
        batch_size, seq_len = batch["actions"].shape[:2]
        flat_obs = {}
        for key, val in batch["observations"].items():
            assert val.shape[0] == batch_size and val.shape[1] == seq_len, (
                f"Observation {key} has unexpected shape {val.shape}, "
                f"expected (batch={batch_size}, time={seq_len}, ...)"
            )
            flat_obs[key] = val.reshape(batch_size * seq_len, *val.shape[2:])

        is_dino_wm = getattr(self.world_model, '_is_dino_wm', False)

        flat_embed = self.world_model.encoder(flat_obs)

        # Cache ViT patch features for feature loss (DA-V2 encoder only)
        cached_vit_features = getattr(
            self.world_model.encoder, '_cached_vit_patch_features', None
        )

        if is_dino_wm:
            # DINO-WM: use raw ViT patch features directly (no projection)
            # cached_vit_features: (B*T, N_vit, D_vit)
            embed = cached_vit_features.reshape(
                batch_size, seq_len, *cached_vit_features.shape[1:]
            )
        else:
            embed = flat_embed.reshape(batch_size, seq_len, -1)

        actions_one_hot = F.one_hot(
            batch["actions"].long().squeeze(-1),
            num_classes=self.world_model.dynamics._num_actions,
        ).float()

        post, prior = self.world_model.dynamics.observe(
            embed, actions_one_hot, batch["is_first"]
        )
        # 1-D pooled feat for reward/traj decoders
        feat = self.world_model.dynamics.get_feat(post)

        # Depth loss
        depth_key = next(
            (k for k in batch["observations"] if "depth" in k.lower()),
            None,
        )
        if is_dino_wm:
            patch_feat = self.world_model.get_patch_feat(post)
            depth_dist = self.world_model.heads["depth"](patch_feat)
        else:
            depth_dist = self.world_model.heads["depth"](feat)

        if depth_key is not None:
            depth_target = batch["observations"][depth_key]
            depth_loss = -depth_dist.log_prob(depth_target).mean()
        else:
            depth_loss = depth_dist.mean().mean() * 0.0

        # Feature prediction loss (must run before depth_pred_loss so that
        # compute_z_loss caches pred_deter for reuse — single _predict pass).
        zero = depth_loss.new_tensor(0.0)
        feature_loss = zero
        if is_dino_wm:
            if feature_loss_scale > 0 and cached_vit_features is not None:
                feature_loss = self.world_model.dynamics.compute_z_loss(
                    post, cached_vit_features, batch_size, seq_len
                )
        else:
            if (feature_loss_scale > 0
                    and "vit_features" in self.world_model.heads
                    and cached_vit_features is not None):
                vit_target = cached_vit_features.detach().reshape(
                    batch_size, seq_len, *cached_vit_features.shape[1:]
                )
                vit_pred = self.world_model.heads["vit_features"](feat)
                feature_loss = F.mse_loss(vit_pred, vit_target)

        # Depth prediction loss: predictor features → DPT → GT depth.
        # get_pred_patch_feat reuses pred_deter cached by compute_z_loss above.
        depth_pred_loss = depth_loss.new_tensor(0.0)
        if is_dino_wm and depth_key is not None and depth_pred_loss_scale > 0:
            pred_patch_feat = self.world_model.get_pred_patch_feat(post)
            pred_depth_dist = self.world_model.heads["depth"](pred_patch_feat)
            depth_pred_loss = -pred_depth_dist.log_prob(
                batch["observations"][depth_key]
            ).mean()

        # Human trajectory loss (force goal path active when enabled).
        traj_head = self.world_model.heads["human_traj"]
        traj_key = next(
            (k for k in batch["observations"] if "future_trajectory" in k.lower()),
            None,
        )
        human_state_goal = next(
            (
                batch["observations"][k]
                for k in batch["observations"]
                if "human_state_goal" in k
            ),
            None,
        )
        if human_state_goal is None and getattr(
            traj_head, "use_goal_conditioning", False
        ):
            human_state_goal = feat.new_zeros(
                batch_size,
                seq_len,
                traj_head.num_humans,
                traj_head.state_goal_dim,
            )
        traj_dist = traj_head(
            feat,
            human_state_goal=human_state_goal,
        )
        traj_mean = traj_dist.mean()
        if traj_key is not None:
            traj_target = batch["observations"][traj_key]
            num_humans = traj_target.shape[2]
            human_num_key = next(
                (k for k in batch["observations"] if "human_num" in k.lower()),
                None,
            )
            if human_num_key is not None:
                human_num = (
                    batch["observations"][human_num_key]
                    .squeeze(-1)
                    .long()
                    .clamp(0, num_humans)
                )
                mask = (
                    torch.arange(num_humans, device=traj_target.device)
                    < human_num.unsqueeze(-1)
                ).float()
                # Per-human mean-squared error, masked by valid humans
                sq = (traj_mean - traj_target).pow(2).mean(dim=(-2, -1))
                traj_loss = (sq * mask).sum() / mask.sum().clamp(min=1e-8)
            else:
                traj_loss = -traj_dist.log_prob(traj_target).mean()
        else:
            traj_loss = traj_mean.mean() * 0.0

        # Reward loss.
        reward_dist = self.world_model.heads["reward"](feat)
        reward_target = batch["rewards"].float().reshape(batch_size, seq_len, -1)
        if reward_target.dim() == 2:
            reward_target = reward_target.unsqueeze(-1)
        reward_loss = -reward_dist.log_prob(reward_target).mean()

        # KL with DreamerV3-style balancing.
        kl_loss, kl_value, dyn_loss, rep_loss = self.world_model.dynamics.kl_loss(
            post, prior,
            free=kl_free_bits,
            dyn_scale=kl_dyn_scale,
            rep_scale=kl_rep_scale,
        )
        kl_loss = kl_loss.mean()

        # --- Imagination loss: multi-step prior supervision ---
        zero = depth_loss.new_tensor(0.0)
        imag_depth_loss = zero
        imag_reward_loss = zero
        imag_traj_loss = zero

        if imag_loss_scale > 0 and seq_len >= 4:
            mid_t = seq_len // 2
            max_H = seq_len - mid_t - 1
            H = min(imag_loss_horizon, max_H)

            # Extract human_state_goal at mid_t (held constant during rollout)
            hsg_mid = (
                human_state_goal[:, mid_t]
                if human_state_goal is not None
                else None
            )

            state = {k: v[:, mid_t] for k, v in post.items()}
            gamma = imag_loss_horizon_gamma
            weight_sum = 0.0
            acc_depth, acc_reward, acc_traj = zero, zero, zero

            if hasattr(self.world_model.dynamics, 'reset_imagination'):
                self.world_model.dynamics.reset_imagination()
            if hasattr(self.world_model.dynamics, 'reset_kv_cache'):
                self.world_model.dynamics.reset_kv_cache(batch_size)

            for h in range(H):
                action_h = actions_one_hot[:, mid_t + h]
                state = self.world_model.dynamics.img_step(state, action_h)
                imag_feat = self.world_model.dynamics.get_feat(state)
                w = gamma ** h
                weight_sum += w
                target_t = mid_t + 1 + h

                # Imagination depth loss
                if depth_key is not None and imag_loss_depth_scale > 0:
                    if is_dino_wm:
                        imag_patch_feat = self.world_model.get_patch_feat(state)
                        d_dist = self.world_model.heads["depth"](imag_patch_feat)
                    else:
                        d_dist = self.world_model.heads["depth"](imag_feat)
                    acc_depth = acc_depth + w * (
                        -d_dist.log_prob(depth_target[:, target_t]).mean()
                    )

                # Imagination reward loss
                if imag_loss_reward_scale > 0:
                    r_dist = self.world_model.heads["reward"](imag_feat)
                    acc_reward = acc_reward + w * (
                        -r_dist.log_prob(reward_target[:, target_t]).mean()
                    )

                # Imagination trajectory loss
                if traj_key is not None and imag_loss_traj_scale > 0:
                    t_dist = self.world_model.heads["human_traj"](
                        imag_feat, human_state_goal=hsg_mid,
                    )
                    t_mean = t_dist.mean()
                    t_target = traj_target[:, target_t]
                    if human_num_key is not None:
                        h_num = (
                            batch["observations"][human_num_key][:, target_t]
                            .squeeze(-1).long().clamp(0, num_humans)
                        )
                        h_mask = (
                            torch.arange(num_humans, device=t_target.device)
                            < h_num.unsqueeze(-1)
                        ).float()
                        sq_h = (t_mean - t_target).pow(2).mean(dim=(-2, -1))
                        acc_traj = acc_traj + w * (
                            (sq_h * h_mask).sum() / h_mask.sum().clamp(min=1e-8)
                        )
                    else:
                        acc_traj = acc_traj + w * (
                            -t_dist.log_prob(t_target).mean()
                        )

            if weight_sum > 0:
                imag_depth_loss = acc_depth / weight_sum
                imag_reward_loss = acc_reward / weight_sum
                imag_traj_loss = acc_traj / weight_sum

        return {
            "depth_loss": depth_loss,
            "depth_pred_loss": depth_pred_loss,
            "traj_loss": traj_loss,
            "reward_loss": reward_loss,
            "kl_loss": kl_loss,
            "feature_loss": feature_loss,
            "imag_depth_loss": imag_depth_loss,
            "imag_reward_loss": imag_reward_loss,
            "imag_traj_loss": imag_traj_loss,
        }


@baseline_registry.register_trainer(name="navthinker_trainer")
class NavThinkerTrainer(BaseRLTrainer):
    r"""Trainer class for NavThinker algorithm
    """
    supported_tasks = ["Nav-v0"]

    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    _is_distributed: bool
    envs: VectorEnv
    _env_spec: Optional[EnvironmentSpec]

    def __init__(self, config=None):
        super().__init__(config)

        self._agent = None
        self.envs = None
        self.obs_transforms = []
        self._is_static_encoder = False
        self._encoder = None
        self._env_spec = None

        # Distributed if the world size would be
        # greater than 1
        self._is_distributed = get_distrib_size()[2] > 1

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        r"""All reduce helper method that moves things to the correct
        device and only runs if distributed
        """
        if not self._is_distributed:
            return t

        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)

    def _create_obs_transforms(self):
        self.obs_transforms = get_active_obs_transforms(self.config)
        self._env_spec.observation_space = apply_obs_transforms_obs_space(
            self._env_spec.observation_space, self.obs_transforms
        )

    def _create_agent(self, resume_state, **kwargs) -> AgentAccessMgr:
        """
        Sets up the AgentAccessMgr. You still must call `agent.post_init` after
        this call. This only constructs the object.
        """

        self._create_obs_transforms()
        return baseline_registry.get_agent_access_mgr(
            self.config.habitat_baselines.rl.agent.type
        )(
            config=self.config,
            env_spec=self._env_spec,
            is_distrib=self._is_distributed,
            device=self.device,
            resume_state=resume_state,
            num_envs=self.envs.num_envs,
            percent_done_fn=self.percent_done,
            **kwargs,
        )

    def _init_envs(self, config=None, is_eval: bool = False):
        if config is None:
            config = self.config
        # print(config) ##
        env_factory: VectorEnvFactory = hydra.utils.instantiate(
            config.habitat_baselines.vector_env_factory
        )
        self.envs = env_factory.construct_envs(
            config,
            workers_ignore_signals=is_slurm_batch_job(),
            enforce_scenes_greater_eq_environments=is_eval,
            is_first_rank=(
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ),
        )

        self._env_spec = EnvironmentSpec(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            orig_action_space=self.envs.orig_action_spaces[0],
        )

        # The measure keys that should only be logged on rank0 and nowhere
        # else. They will be excluded from all other workers and only reported
        # from the single worker.
        self._rank0_keys: Set[str] = set(
            list(self.config.habitat.task.rank0_env0_measure_names)
            + list(self.config.habitat.task.rank0_measure_names)
        )

        # Information on measures that declared in `self._rank0_keys` or
        # to be only reported on rank0. This is seperately logged from
        # `self.window_episode_stats`.
        self._single_proc_infos: Dict[str, List[float]] = {}

    def _init_train(self, resume_state=None):
        if resume_state is None:
            resume_state = load_resume_state(self.config)

        if resume_state is not None:
            if not self.config.habitat_baselines.load_resume_state_config:
                raise FileExistsError(
                    f"The configuration provided has habitat_baselines.load_resume_state_config=False but a previous training run exists. You can either delete the checkpoint folder {self.config.habitat_baselines.checkpoint_folder}, or change the configuration key habitat_baselines.checkpoint_folder in your new run."
                )

            self.config = self._get_resume_state_config_or_new_config(
                resume_state["config"]
            )

        if self.config.habitat_baselines.rl.ddppo.force_distributed:
            self._is_distributed = True

        self._add_preemption_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.habitat_baselines.rl.ddppo.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            with read_write(self.config):
                self.config.habitat_baselines.torch_gpu_id = local_rank
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
                    local_rank
                )
                # Multiply by the number of simulators to make sure they also get unique seeds
                self.config.habitat.seed += (
                    torch.distributed.get_rank()
                    * self.config.habitat_baselines.num_environments
                )

            random.seed(self.config.habitat.seed)
            np.random.seed(self.config.habitat.seed)
            torch.manual_seed(self.config.habitat.seed)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.habitat_baselines.verbose:
            logger.info(f"config: {OmegaConf.to_yaml(self.config)}")

        profiling_wrapper.configure(
            capture_start_step=self.config.habitat_baselines.profiling.capture_start_step,
            num_steps_to_capture=self.config.habitat_baselines.profiling.num_steps_to_capture,
        )

        # remove the non scalar measures from the measures since they can only be used in
        # evaluation.  In collect-only mode, keep top_down_map for video generation.
        _wm_cfg = self.config.habitat_baselines.get("world_model", None)
        _keep_topdown = (
            _wm_cfg is not None
            and getattr(_wm_cfg, "collect_replay_only", False)
        )
        for non_scalar_metric in NON_SCALAR_METRICS:
            non_scalar_metric_root = non_scalar_metric.split(".")[0]
            if _keep_topdown and non_scalar_metric_root == "top_down_map":
                continue
            if non_scalar_metric_root in self.config.habitat.task.measurements:
                with read_write(self.config):
                    OmegaConf.set_struct(self.config, False)
                    self.config.habitat.task.measurements.pop(
                        non_scalar_metric_root
                    )
                    OmegaConf.set_struct(self.config, True)
                if self.config.habitat_baselines.verbose:
                    logger.info(
                        f"Removed metric {non_scalar_metric_root} from metrics since it cannot be used during training."
                    )

        self._init_envs()

        # In collect-only mode, RGB keys are in obs_keys so the env returns
        # them, but they must NOT enter the policy / visual encoder / rollout
        # storage.  Strip them from _env_spec.observation_space here so that
        # downstream components (agent, rollout storage, visual encoder) never
        # see them.  The raw RGB data is cached separately in
        # _collect_environment_result for video generation.
        wm_cfg = self.config.habitat_baselines.get("world_model", None)
        self._collect_visual_strip_keys: List[str] = []
        if wm_cfg is not None and getattr(wm_cfg, "collect_replay_only", False):
            from gym import spaces as gym_spaces
            obs_sp = self._env_spec.observation_space
            strip_keys = [
                k for k in obs_sp.spaces
                if "rgb" in k and len(obs_sp.spaces[k].shape) > 1
            ]
            if strip_keys:
                self._collect_visual_strip_keys = strip_keys
                new_spaces = {
                    k: v for k, v in obs_sp.spaces.items()
                    if k not in strip_keys
                }
                self._env_spec = EnvironmentSpec(
                    observation_space=gym_spaces.Dict(new_spaces),
                    action_space=self._env_spec.action_space,
                    orig_action_space=self._env_spec.orig_action_space,
                )
                if rank0_only():
                    logger.info(
                        f"[Collect] Stripped RGB keys from observation_space "
                        f"(policy/storage will not see them): {strip_keys}"
                    )

        self.device = get_device(self.config)

        if rank0_only() and not os.path.isdir(
            self.config.habitat_baselines.checkpoint_folder
        ):
            os.makedirs(self.config.habitat_baselines.checkpoint_folder)

        logger.add_filehandler(self.config.habitat_baselines.log_file)

        self._agent = self._create_agent(resume_state)
        if self._is_distributed:
            self._agent.init_distributed(find_unused_params=False)  # type: ignore
        self._agent.post_init()

        self._is_static_encoder = (
            not self.config.habitat_baselines.rl.ddppo.train_encoder
        )
        self._ppo_cfg = self.config.habitat_baselines.rl.ppo

        observations = self.envs.reset()
        observations = self.envs.post_step(observations)
        # Strip RGB keys from initial observations (collect mode)
        if self._collect_visual_strip_keys:
            for obs_dict in observations:
                for vk in self._collect_visual_strip_keys:
                    obs_dict.pop(vk, None)
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        # Key Modification between the trainer and the original ppo trainer
        if self._is_static_encoder:
            self._encoder = self._agent.actor_critic.visual_encoder
            if self._encoder is None:
                self._encoder = self._agent._agents[0].actor_critic.visual_encoder
                with inference_mode():
                    batch_temp = {key.replace('agent_0_', ''): value for key, value in batch.items()}
                    batch[
                        'agent_0_' + PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch_temp)
            else:
                with inference_mode():
                    batch[
                        PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch)

        self._agent.rollouts.insert_first_observations(batch)

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=self._ppo_cfg.reward_window_size)
        )
        self._resume_update_num = 0
        # logger.info(f"[INIT DEBUG] running_episode_stats initialized with "
        #              f"num_envs={self.envs.num_envs}, all zeros. "
        #              f"window_size={self._ppo_cfg.reward_window_size}")

        # ==================== World Model Training Setup ====================
        self._init_world_model_training()

        # Load WM optimizer state if resuming
        if resume_state is not None and self.train_world_model:
            self._load_wm_state(resume_state)

        self.t_start = time.time()

    def _init_world_model_training(self):
        """Initialize World Model training components"""
        # Default so that early returns still leave attribute defined (avoids AttributeError in training loop).
        self.imagine_enabled = False
        # Check if WM is enabled (supports both dict-like and structured WorldModelConfig)
        wm_config = getattr(self.config.habitat_baselines, "world_model", None)
        if wm_config is None:
            self.use_world_model = False
            self.train_world_model = False
        else:
            self.use_world_model = getattr(wm_config, "enabled", False)
            self.train_world_model = self.use_world_model and getattr(
                wm_config, "train_world_model", False
            )

        # Collect-only mode: store rollouts to replay buffer but skip WM training.
        # Must be checked before the early return so that collect mode works
        # even when the policy does not use a world model (enabled: False).
        self.collect_replay_only = getattr(wm_config, "collect_replay_only", False) if wm_config is not None else False

        if not self.use_world_model and not self.collect_replay_only:
            logger.info("World Model is disabled")
            return

        if not self.use_world_model and self.collect_replay_only:
            logger.info("World Model disabled in policy (enabled: False), collect-replay-only mode")
        else:
            # Get WM from policy (needed for both training and pretrained loading)
            try:
                if hasattr(self._agent.actor_critic, 'net'):
                    self.world_model = getattr(self._agent.actor_critic.net, 'world_model', None)
                else:
                    self.world_model = getattr(self._agent._agents[0].actor_critic.net, 'world_model', None)

                if self.world_model is None:
                    logger.warning("World Model not found in policy, disabling WM training")
                    self.train_world_model = False
                    if not self.collect_replay_only:
                        return
            except Exception as e:
                logger.warning(f"Failed to get World Model: {e}, disabling WM training")
                self.train_world_model = False
                if not self.collect_replay_only:
                    return

            # Load pretrained World Model weights if specified
            pretrained_wm_path = getattr(wm_config, "pretrained_wm_checkpoint", None)
            if pretrained_wm_path is not None and pretrained_wm_path != "":
                self._load_pretrained_world_model(pretrained_wm_path)

        if not self.train_world_model and not self.collect_replay_only:
            self.imagine_horizon = getattr(wm_config, "imagine_horizon", 15)
            self.imagine_social_coeff = getattr(wm_config, "imagine_social_coeff", 0.1)
            self.imagine_safety_margin = getattr(wm_config, "imagine_safety_margin", 0.5)
            self.imagine_intrinsic_coeff = getattr(wm_config, "imagine_intrinsic_coeff", 0.01)
            self.imagine_enabled = getattr(wm_config, "imagine_enabled", False)
            logger.info("World Model training is disabled (pretrained WM loaded for inference only)")
            if self.imagine_enabled:
                logger.info(f"  - Social imagination ENABLED (frozen WM)")
                logger.info(f"  - imagine_horizon: {self.imagine_horizon}")
                logger.info(f"  - imagine_social_coeff: {self.imagine_social_coeff}")
                logger.info(f"  - imagine_safety_margin: {self.imagine_safety_margin}")
                logger.info(f"  - imagine_intrinsic_coeff: {self.imagine_intrinsic_coeff}")
            return

        if self.collect_replay_only and not self.train_world_model:
            logger.info("Collect-replay-only mode: episodes saved to disk individually")
            buffer_size = getattr(wm_config, "replay_buffer_size", 100000)
            self.replay_buffer_size = buffer_size
            self.replay_buffer = None
            self.replay_buffer_num_envs = None
            self.replay_buffer_env_capacity = None
            self.replay_buffer_warmup = 0
            self.replay_buffer_path = getattr(wm_config, "replay_buffer_path", None) or ""
            self.replay_buffer_save_interval = getattr(wm_config, "replay_buffer_save_interval", 50)
            self.collect_replay_episodes = getattr(wm_config, "collect_replay_episodes", 0)
            self.wm_vis_interval = 0
            self.wm_offline_only = False
            self.wm_offline_updates = 0
            self.imagine_enabled = False
            self._episode_steps = None
            self._episode_write_counter = 0
            self._collect_episodes_written = 0
            self.save_failed_episodes = getattr(wm_config, "save_failed_episodes", False)

            self.save_depth_images = getattr(wm_config, "save_depth_images", True)
            self.save_depth_video = getattr(wm_config, "save_depth_video", True)
            self.save_rgb_images = getattr(wm_config, "save_rgb_images", True)
            self.save_rgb_video = getattr(wm_config, "save_rgb_video", True)
            self.save_third_rgb_images = getattr(wm_config, "save_third_rgb_images", True)
            self.save_third_rgb_video = getattr(wm_config, "save_third_rgb_video", True)
            self.save_topdown_images = getattr(wm_config, "save_topdown_images", True)
            self.save_combined_video = getattr(wm_config, "save_combined_video", True)
            self.save_combined_images = getattr(wm_config, "save_combined_images", False)

            self._envs_exhausted = set()
            self._all_envs_exhausted = False

            ep_dir = self._get_episode_dir()
            logger.info(f"  - episode_dir: {ep_dir}")
            logger.info(f"  - replay_buffer_path: {self.replay_buffer_path or '(none)'}")
            if self.collect_replay_episodes > 0:
                logger.info(f"  - collect_replay_episodes: {self.collect_replay_episodes}")
            else:
                logger.info(f"  - collect_replay_episodes: unlimited (controlled by total_num_steps)")
            logger.info(f"  - save_failed_episodes: {self.save_failed_episodes}")
            logger.info(
                f"  - save visuals: depth_img={self.save_depth_images} depth_vid={self.save_depth_video} "
                f"rgb_img={self.save_rgb_images} rgb_vid={self.save_rgb_video} "
                f"third_img={self.save_third_rgb_images} third_vid={self.save_third_rgb_video} "
                f"topdown_img={self.save_topdown_images} combined_vid={self.save_combined_video}"
            )
            return

        logger.info("Initializing World Model training...")

        # Freeze WM encoder if specified (after loading pretrained weights)
        self.freeze_wm_encoder = getattr(wm_config, "freeze_wm_encoder", False)
        if self.freeze_wm_encoder:
            self.world_model.freeze_encoder()

        # WM training parameters (from config)
        self.wm_train_ratio = getattr(wm_config, "wm_train_ratio", 0.1)
        self.wm_warmup_updates = getattr(wm_config, "wm_warmup_updates", 1000)
        self.wm_grad_clip = getattr(wm_config, "wm_grad_clip", 100.0)
        self.wm_batch_size = getattr(wm_config, "wm_batch_size", 16)
        self.wm_sequence_length = getattr(wm_config, "wm_sequence_length", 50)
        self.wm_epochs_per_update = getattr(wm_config, "wm_epochs_per_update", 1)

        # Loss scales
        self.depth_loss_scale = getattr(wm_config, "depth_loss_scale", 1.0)
        self.depth_pred_loss_scale = getattr(wm_config, "depth_pred_loss_scale", 0.0)
        self.traj_loss_scale = getattr(wm_config, "traj_loss_scale", 1.0)
        self.reward_loss_scale = getattr(wm_config, "reward_loss_scale", 1.0)
        self.kl_loss_scale = getattr(wm_config, "kl_loss_scale", 0.1)
        self.kl_free_bits = getattr(wm_config, "kl_free_bits", 1.0)
        self.kl_dyn_scale = getattr(wm_config, "kl_dyn_scale", 0.5)
        self.kl_rep_scale = getattr(wm_config, "kl_rep_scale", 0.1)

        # Create WM optimizer (only include trainable params, excludes frozen encoder)
        self.wm_optimizer = torch.optim.Adam(
            self.world_model.trainable_parameters(),
            lr=getattr(wm_config, "wm_lr", 3e-4),
            eps=getattr(wm_config, "opt_eps", 1e-5),
            weight_decay=getattr(wm_config, "weight_decay", 0.0),
        )

        # Build WM train wrapper and optionally wrap with DDP.
        self.wm_train_model = _WorldModelTrainModule(self.world_model).to(
            self.device
        )
        self.wm_ddp_enabled = False
        wm_use_ddp = getattr(wm_config, "ddp", True)
        if self._is_distributed and wm_use_ddp:
            if self.device.type == "cuda":
                device_index = (
                    self.device.index if self.device.index is not None else 0
                )
                self.wm_train_model = torch.nn.parallel.DistributedDataParallel(
                    self.wm_train_model,
                    device_ids=[device_index],
                    output_device=device_index,
                    find_unused_parameters=False,
                )
            else:
                self.wm_train_model = torch.nn.parallel.DistributedDataParallel(
                    self.wm_train_model,
                    find_unused_parameters=False,
                )
            self.wm_ddp_enabled = True

        # Imagination config
        self.imagine_horizon = getattr(wm_config, "imagine_horizon", 15)
        self.imagine_social_coeff = getattr(wm_config, "imagine_social_coeff", 0.1)
        self.imagine_safety_margin = getattr(wm_config, "imagine_safety_margin", 0.5)
        self.imagine_intrinsic_coeff = getattr(wm_config, "imagine_intrinsic_coeff", 0.01)
        self.imagine_enabled = getattr(wm_config, "imagine_enabled", False)

        # Imagination loss config (multi-step prior supervision)
        self.imag_loss_scale = getattr(wm_config, "imag_loss_scale", 0.0)
        self.imag_loss_depth_scale = getattr(wm_config, "imag_loss_depth_scale", 1.0)
        self.imag_loss_reward_scale = getattr(wm_config, "imag_loss_reward_scale", 1.0)
        self.imag_loss_traj_scale = getattr(wm_config, "imag_loss_traj_scale", 1.0)
        self.imag_loss_horizon_gamma = getattr(wm_config, "imag_loss_horizon_gamma", 0.95)
        self.imag_loss_horizon = getattr(wm_config, "imag_loss_horizon", 5)

        # Feature loss config (DA-V2 ViT feature prediction)
        self.feature_loss_scale = getattr(wm_config, "feature_loss_scale", 0.0)

        # Create replay buffer (env-wise trajectories, initialized lazily)
        buffer_size = getattr(wm_config, "replay_buffer_size", 100000)
        self.replay_buffer_size = buffer_size
        self.replay_buffer = None
        self.replay_buffer_num_envs = None
        self.replay_buffer_env_capacity = None
        self.replay_buffer_warmup = getattr(wm_config, "replay_buffer_warmup", 5000)
        self.wm_vis_interval = getattr(wm_config, "wm_vis_interval", 20)

        # Replay buffer 独立存储路径 + 离线训练模式
        self.replay_buffer_path = getattr(wm_config, "replay_buffer_path", None) or ""
        self.replay_buffer_save_interval = getattr(wm_config, "replay_buffer_save_interval", 50)
        self.wm_offline_only = getattr(wm_config, "wm_offline_only", False)
        self.wm_offline_updates = getattr(wm_config, "wm_offline_updates", 10000)

        logger.info(f"World Model training initialized:")
        logger.info(f"  - wm_train_ratio: {self.wm_train_ratio} (update every {int(1/self.wm_train_ratio)} policy updates)")
        logger.info(f"  - wm_warmup_updates: {self.wm_warmup_updates}")
        logger.info("  - wm_mode: decoupled (independent optimizer)")
        logger.info(f"  - freeze_wm_encoder: {self.freeze_wm_encoder}")
        logger.info(f"  - replay_buffer_size: {buffer_size}")
        logger.info(f"  - wm_vis_interval: {self.wm_vis_interval} (0=disable)")
        logger.info(f"  - wm_batch_size: {self.wm_batch_size}")
        logger.info(f"  - wm_sequence_length: {self.wm_sequence_length}")
        logger.info(f"  - wm_ddp: {self.wm_ddp_enabled}")
        logger.info(f"  - replay_buffer_path: {self.replay_buffer_path or '(none)'}")
        logger.info(f"  - replay_buffer_save_interval: {self.replay_buffer_save_interval}")
        logger.info(f"  - wm_offline_only: {self.wm_offline_only}")
        if self.wm_offline_only:
            logger.info(f"  - wm_offline_updates: {self.wm_offline_updates}")
        logger.info(f"  - imagine_enabled: {self.imagine_enabled}")
        if self.imagine_enabled:
            logger.info(f"  - imagine_horizon: {self.imagine_horizon}")
            logger.info(f"  - imagine_social_coeff: {self.imagine_social_coeff}")
            logger.info(f"  - imagine_safety_margin: {self.imagine_safety_margin}")
            logger.info(f"  - imagine_intrinsic_coeff: {self.imagine_intrinsic_coeff}")

        # WM is trained only by wm_optimizer, never by PPO loss.
        self._set_world_model_grad_enabled(False)
        self._clear_world_model_grads()

    def _set_world_model_grad_enabled(self, enabled: bool) -> None:
        if not hasattr(self, "world_model") or self.world_model is None:
            return
        skip_encoder = getattr(self, "freeze_wm_encoder", False)
        encoder_params = (
            set(self.world_model.encoder.parameters())
            if skip_encoder and hasattr(self.world_model, "encoder")
            else set()
        )
        for param in self.world_model.parameters():
            if param not in encoder_params:
                param.requires_grad_(enabled)

    def _clear_world_model_grads(self) -> None:
        if not hasattr(self, "world_model") or self.world_model is None:
            return
        if hasattr(self, "wm_optimizer") and self.wm_optimizer is not None:
            self.wm_optimizer.zero_grad(set_to_none=True)
        for param in self.world_model.parameters():
            param.grad = None

    def _load_pretrained_world_model(self, checkpoint_path: str):
        """加载预训练的 World Model 权重

        Args:
            checkpoint_path: 预训练checkpoint的路径（可以是完整checkpoint或只包含world_model的文件）
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Pretrained WM checkpoint not found: {checkpoint_path}")
            return

        try:
            if rank0_only():
                logger.info(f"Loading pretrained World Model from: {checkpoint_path}")

            # Load checkpoint（与 save_checkpoint 一致：单 agent 为 state_dict，多 agent 为 {0: {state_dict}, 1: {...}}）
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict")
            if state_dict is None:
                for v in checkpoint.values():
                    if isinstance(v, dict) and "state_dict" in v:
                        state_dict = v["state_dict"]
                        break
            # state_dict 来自 policy.state_dict()，key 为 "net.world_model.xxx"（单 agent）或带 "module."（DDP）
            wm_prefixes = ("net.world_model.", "module.net.world_model.")
            wm_state_dict = {}
            if state_dict:
                for k, v in state_dict.items():
                    if not isinstance(k, str):
                        continue
                    for prefix in wm_prefixes:
                        if k.startswith(prefix):
                            wm_state_dict[k[len(prefix):]] = v
                            break

            # Also try direct world_model_state_dict (from offline WM training)
            if not wm_state_dict and "world_model_state_dict" in checkpoint:
                wm_state_dict = checkpoint["world_model_state_dict"]

            # Standalone offline checkpoint format: separate dynamics / depth_head / vit dicts
            if not wm_state_dict and "dynamics_state_dict" in checkpoint:
                for k, v in checkpoint.get("dynamics_state_dict", {}).items():
                    wm_state_dict[f"dynamics.{k}"] = v
                for k, v in checkpoint.get("depth_head_state_dict", {}).items():
                    wm_state_dict[f"heads.depth.{k}"] = v
                for k, v in checkpoint.get("vit_state_dict", {}).items():
                    wm_state_dict[f"encoder._vit.{k}"] = v
                if rank0_only():
                    logger.info(
                        f"  Assembled WM state_dict from standalone offline checkpoint: "
                        f"dynamics={len(checkpoint.get('dynamics_state_dict', {}))}, "
                        f"depth_head={len(checkpoint.get('depth_head_state_dict', {}))}, "
                        f"vit={len(checkpoint.get('vit_state_dict', {}))}"
                    )

            if not wm_state_dict:
                logger.warning("No world_model parameters found in checkpoint")
                return

            # Load weights
            missing_keys, unexpected_keys = self.world_model.load_state_dict(
                wm_state_dict, strict=False
            )

            if rank0_only():
                loaded_keys = [k for k in wm_state_dict if k not in missing_keys]
                loaded_params = sum(wm_state_dict[k].numel() for k in loaded_keys)
                total_params = sum(p.numel() for p in self.world_model.parameters())
                logger.info(
                    f"[World Model] Successfully loaded pretrained WM checkpoint!\n"
                    f"  Source: {checkpoint_path}\n"
                    f"  Loaded layers: {len(loaded_keys)}/{len(wm_state_dict)}\n"
                    f"  Loaded params: {loaded_params:,} / {total_params:,} "
                    f"({loaded_params/total_params*100:.1f}%)"
                )
                if missing_keys:
                    logger.warning(
                        f"  Missing keys ({len(missing_keys)}, will be randomly initialized):"
                    )
                    for key in missing_keys[:10]:
                        logger.warning(f"    - {key}")
                    if len(missing_keys) > 10:
                        logger.warning(f"    ... and {len(missing_keys) - 10} more")
                if unexpected_keys:
                    logger.warning(
                        f"  Unexpected keys ({len(unexpected_keys)}, ignored):"
                    )
                    for key in unexpected_keys[:10]:
                        logger.warning(f"    - {key}")
                    if len(unexpected_keys) > 10:
                        logger.warning(f"    ... and {len(unexpected_keys) - 10} more")

        except Exception as e:
            logger.error(f"Failed to load pretrained World Model: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _load_wm_state(self, resume_state):
        """从 checkpoint 加载 World Model optimizer 状态 + replay buffer"""
        if 'wm_optimizer_state' in resume_state:
            self.wm_optimizer.load_state_dict(resume_state['wm_optimizer_state'])
            if rank0_only():
                logger.info("Loaded WM optimizer state from checkpoint")

        ckpt_dir = self.config.habitat_baselines.checkpoint_folder
        if self._load_replay_buffer(ckpt_dir):
            if rank0_only():
                logger.info(
                    f"WM replay buffer restored: {self._get_replay_buffer_size()} experiences"
                )
        elif 'replay_buffer_size' in resume_state:
            prev_size = resume_state['replay_buffer_size']
            if rank0_only():
                logger.info(
                    f"WM replay buffer file not found, will re-collect "
                    f"(previous size was {prev_size})"
                )

    @rank0_only
    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            **self._agent.get_save_state(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state  # type: ignore

        # Save World Model optimizer state + replay buffer
        if self.train_world_model and hasattr(self, 'wm_optimizer'):
            checkpoint["wm_optimizer_state"] = self.wm_optimizer.state_dict()
            checkpoint["replay_buffer_size"] = self._get_replay_buffer_size()
            self._save_replay_buffer(
                self.config.habitat_baselines.checkpoint_folder
            )
            logger.info(
                "Saving WM optimizer state "
                f"(replay buffer size: {self._get_replay_buffer_size()})"
            )

        save_file_path = os.path.join(
            self.config.habitat_baselines.checkpoint_folder, file_name
        )
        torch.save(checkpoint, save_file_path)
        torch.save(
            checkpoint,
            os.path.join(
                self.config.habitat_baselines.checkpoint_folder, "latest.pth"
            ),
        )
        if self.config.habitat_baselines.on_save_ckpt_callback is not None:
            hydra.utils.call(
                self.config.habitat_baselines.on_save_ckpt_callback,
                save_file_path=save_file_path,
            )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        kwargs.setdefault("weights_only", False)
        return torch.load(checkpoint_path, *args, **kwargs)

    def _should_update_world_model(self, update):
        """判断是否应该更新 World Model"""
        if not self.train_world_model:
            return False

        # 检查 warmup
        if update < self.wm_warmup_updates:
            return False

        # 检查 replay buffer 是否足够
        if self._get_replay_buffer_size() < self.replay_buffer_warmup:
            return False

        # 检查更新频率（关键逻辑：wm_train_ratio）
        update_interval = int(1 / self.wm_train_ratio)
        if update % update_interval != 0:
            return False

        return True

    def _get_replay_buffer_size(self) -> int:
        """Get total WM replay size (steps) across all episodes / env trajectories."""
        ep_index = getattr(self, "_ep_file_index", None)
        if ep_index is not None:
            return getattr(self, "_ep_total_steps", 0)
        if self.replay_buffer is None:
            return 0
        return int(sum(len(env_buffer) for env_buffer in self.replay_buffer))

    # ---- Episode-level disk storage (collect_replay_only) ----

    def _get_episode_dir(self) -> str:
        """Return the directory where per-episode .pt files are stored."""
        p = getattr(self, "replay_buffer_path", "")
        if p.endswith(".pt"):
            p = os.path.splitext(p)[0]
        return p

    def _flush_episode_to_disk(
        self, steps: list, env_idx: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Write a completed episode (list of step dicts) to disk.

        Each step may carry ``raw_visual`` (original env observations with
        full-resolution RGB) and ``info`` (containing ``top_down_map`` etc.)
        cached by ``_collect_environment_result``.  These are used to produce
        images / videos and then freed before ``.pt`` serialization to limit
        memory usage.

        Returns the saved file path, or empty string on failure.
        """
        if not steps:
            return ""
        ep_dir = self._get_episode_dir()
        os.makedirs(ep_dir, exist_ok=True)

        rank = 0
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()

        ep_idx = getattr(self, "_episode_write_counter", 0)

        suffix = ""
        if metrics:
            parts = []
            if "spl" in metrics:
                parts.append(f"spl{metrics['spl']:.2f}")
            if "success" in metrics:
                parts.append(f"succ{int(metrics['success'])}")
            if "distance_to_goal" in metrics:
                parts.append(f"dtg{metrics['distance_to_goal']:.1f}")
            if "reward" in metrics:
                parts.append(f"R{metrics['reward']:.2f}")
            if parts:
                suffix = "_" + "_".join(parts)
        dataset_suffix = ""
        if metrics:
            scene_id = metrics.get("scene_id")
            episode_id = metrics.get("episode_id")
            if scene_id is not None or episode_id is not None:
                scene_short = (str(scene_id).split("/")[-1].split(".")[0]) if scene_id else ""
                ep_id_str = str(episode_id) if episode_id is not None else ""
                if scene_short or ep_id_str:
                    dataset_suffix = "_" + "scene=" + (scene_short or "?") + "-ep=" + (ep_id_str or "?")

        # --- Skip if same scene+episode already exists in ep_dir ---
        if dataset_suffix:
            scene_ep_key = dataset_suffix.lstrip("_")  # e.g. "scene=XXX-ep=YYY"
            if not hasattr(self, "_flushed_scene_ep_keys"):
                import re
                self._flushed_scene_ep_keys = set()
                _sep_re = re.compile(r"(scene=\S+?-ep=\w+)")
                try:
                    for fn in os.listdir(ep_dir):
                        if fn.endswith(".pt"):
                            m = _sep_re.search(fn)
                            if m:
                                self._flushed_scene_ep_keys.add(m.group(1))
                except OSError:
                    pass

            if scene_ep_key in self._flushed_scene_ep_keys:
                if rank0_only():
                    logger.info(
                        f"Skipping duplicate episode ({scene_ep_key}), "
                        f"already exists in {ep_dir}"
                    )
                return ""
            self._flushed_scene_ep_keys.add(scene_ep_key)

        _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ep_tag = f"ep_r{rank}_{ep_idx:06d}_{_ts}{dataset_suffix}{suffix}"
        ep_path = os.path.join(ep_dir, f"{ep_tag}.pt")

        # --- Identify visual keys ---
        _wm_cfg = self.config.habitat_baselines.get("world_model", None)
        _save_all_robots = (
            _wm_cfg is not None and getattr(_wm_cfg, "save_all_robots", False)
        )
        _vis_num_robots = (
            getattr(_wm_cfg, "num_robots", 1) if _save_all_robots else 1
        )

        storage_obs_keys = list(steps[0]["observations"].keys())
        raw_visual_keys = list(steps[0].get("raw_visual", {}).keys())
        all_obs_keys = list(set(storage_obs_keys + raw_visual_keys))

        _save_depth_img = getattr(self, "save_depth_images", True)
        _save_depth_vid = getattr(self, "save_depth_video", True)
        _save_rgb_img = getattr(self, "save_rgb_images", True)
        _save_rgb_vid = getattr(self, "save_rgb_video", True)
        _save_third_img = getattr(self, "save_third_rgb_images", True)
        _save_third_vid = getattr(self, "save_third_rgb_video", True)

        _save_topdown_img = getattr(self, "save_topdown_images", True)
        _save_combined_img = getattr(self, "save_combined_images", False)

        # Build per-robot visual key lists
        _robot_vis_keys = []
        for ri in range(_vis_num_robots):
            prefix = f"agent_{ri}_"
            depth_key = None
            for k in all_obs_keys:
                if k.startswith(prefix) and "depth" in k:
                    depth_key = k
                    break
            if depth_key is None:
                for k in all_obs_keys:
                    if "depth" in k and not k.startswith("agent_"):
                        depth_key = k
                        break
            rgb_key = None
            third_rgb_key = None
            for k in all_obs_keys:
                if k.startswith(prefix) and "third_rgb" in k and third_rgb_key is None:
                    third_rgb_key = k
                elif k.startswith(prefix) and "rgb" in k and rgb_key is None:
                    rgb_key = k
            _robot_vis_keys.append({
                "depth": depth_key, "rgb": rgb_key, "third_rgb": third_rgb_key,
                "prefix": prefix, "tag": f"robot{ri}",
            })

        _need_vis_dir = (
            any(rv["depth"] and _save_depth_img for rv in _robot_vis_keys)
            or any(rv["rgb"] and _save_rgb_img for rv in _robot_vis_keys)
            or any(rv["third_rgb"] and _save_third_img for rv in _robot_vis_keys)
            or (_save_topdown_img and any(s.get("info", {}).get("top_down_map") for s in steps))
            or _save_combined_img
        )
        vis_dir = os.path.join(ep_dir, ep_tag)
        if _need_vis_dir:
            os.makedirs(vis_dir, exist_ok=True)

        for rv in _robot_vis_keys:
            tag = rv["tag"]
            sub_dir = os.path.join(vis_dir, tag) if _vis_num_robots > 1 else vis_dir
            if _need_vis_dir and _vis_num_robots > 1:
                os.makedirs(sub_dir, exist_ok=True)

            # Depth
            if rv["depth"]:
                if _save_depth_img:
                    self._save_obs_images(steps, rv["depth"], sub_dir, "depth", is_depth=True)
                if _save_depth_vid:
                    self._save_obs_video(steps, rv["depth"], ep_dir, f"{ep_tag}_{tag}_depth", is_depth=True)

            # RGB
            if rv["rgb"]:
                short = rv["rgb"].replace(rv["prefix"], "")
                if _save_rgb_img:
                    self._save_obs_images(steps, rv["rgb"], sub_dir, short)
                if _save_rgb_vid:
                    self._save_obs_video(steps, rv["rgb"], ep_dir, f"{ep_tag}_{tag}_{short}")

            # Third-person RGB
            if rv["third_rgb"]:
                short = rv["third_rgb"].replace(rv["prefix"], "")
                if _save_third_img:
                    self._save_obs_images(steps, rv["third_rgb"], sub_dir, short)
                if _save_third_vid:
                    self._save_obs_video(steps, rv["third_rgb"], ep_dir, f"{ep_tag}_{tag}_{short}", resize=(512, 512))

        # --- Top-down map images ---
        if _save_topdown_img and _need_vis_dir:
            self._save_topdown_images(steps, vis_dir, num_robots=_vis_num_robots)

        # --- Combined eval-style video + per-frame images ---
        _save_combined = getattr(self, "save_combined_video", True)
        _save_combined_img = getattr(self, "save_combined_images", False)
        if _save_combined or _save_combined_img:
            for ri in range(_vis_num_robots):
                _prefix = f"agent_{ri}_" if _vis_num_robots > 1 else None
                _rtag = f"{ep_tag}_robot{ri}" if _vis_num_robots > 1 else ep_tag
                self._save_combined_video_and_images(
                    steps, ep_dir, _rtag,
                    vis_dir if _need_vis_dir else None,
                    metrics=metrics,
                    save_video=_save_combined,
                    save_images=_save_combined_img,
                    num_robots=_vis_num_robots,
                    agent_prefix=_prefix,
                )

        # --- Free raw_visual / info to reduce memory before .pt serialization ---
        for step in steps:
            step.pop("raw_visual", None)
            step.pop("info", None)

        # --- Save .pt (observations, actions, rewards, masks only) ---
        obs_space_meta = {}
        for k, v in steps[0]["observations"].items():
            obs_space_meta[k] = {"shape": tuple(v.shape), "dtype": str(v.dtype)}
        action_meta = {
            "shape": tuple(steps[0]["actions"].shape),
            "dtype": str(steps[0]["actions"].dtype),
        }

        try:
            pt_steps = []
            for step in steps:
                pt_step = {
                    "observations": step["observations"],
                    "actions": step["actions"],
                    "rewards": step["rewards"],
                    "masks": step["masks"],
                    "is_first": step["is_first"],
                }
                pt_steps.append(pt_step)
            save_dict = {
                "steps": pt_steps,
                "env_idx": env_idx,
                "num_steps": len(pt_steps),
                "obs_space_meta": obs_space_meta,
                "action_meta": action_meta,
                "metrics": metrics or {},
            }
            if metrics:
                if metrics.get("episode_id") is not None:
                    save_dict["episode_id"] = metrics["episode_id"]
                if metrics.get("scene_id") is not None:
                    save_dict["scene_id"] = metrics["scene_id"]
            torch.save(save_dict, ep_path)
        except Exception as e:
            logger.error(f"Failed to save episode {ep_path}: {e}")
            return ""

        self._episode_write_counter = ep_idx + 1
        if rank0_only():
            pt_size_mb = os.path.getsize(ep_path) / (1024 * 1024)
            logger.info(
                f"[Collect] Saved episode {ep_tag} "
                f"({len(steps)} steps, .pt={pt_size_mb:.1f}MB) -> {ep_dir}"
            )
        return ep_path

    # -------------------- Visual saving helpers --------------------

    @staticmethod
    def _save_combined_video_and_images(
        steps: list, out_dir: str, ep_tag: str,
        img_dir: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None, fps: int = 10,
        save_video: bool = True, save_images: bool = False,
        num_robots: int = 1,
        agent_prefix: Optional[str] = None,
    ) -> str:
        """Create eval-style combined video and/or per-frame PNG images.

        Layout: [RGB | GT Depth | Third RGB | TopDown+Traj]

        When *agent_prefix* is set (e.g. ``"agent_1_"``), only observation
        keys starting with that prefix are used, producing a per-robot video.
        """
        video_path = os.path.join(out_dir, f"{ep_tag}_combined.mp4")
        try:
            import cv2
            import imageio
            from PIL import Image
            from habitat.utils.visualizations import maps
            from habitat_baselines.utils.wm_visualizer import (
                compose_unified_frame,
            )

            if save_images and img_dir:
                os.makedirs(img_dir, exist_ok=True)

            writer = None
            pad_h = pad_w = 0
            wrote_any = False

            for i, step in enumerate(steps):
                raw_vis = step.get("raw_visual", {})
                obs_src = raw_vis if raw_vis else step.get("observations", {})

                rgb_obs = None
                gt_depth_obs = None
                third_rgb = None
                for k, v in obs_src.items():
                    if agent_prefix and not k.startswith(agent_prefix):
                        continue
                    val = v
                    if isinstance(val, torch.Tensor):
                        val = val.cpu().numpy()
                    if "third" in k.lower() and "rgb" in k.lower() and val.ndim >= 2:
                        if third_rgb is None:
                            third_rgb = val
                    elif "rgb" in k.lower() and val.ndim >= 2:
                        if rgb_obs is None:
                            rgb_obs = val
                    elif "depth" in k.lower() and val.ndim >= 2:
                        if gt_depth_obs is None:
                            gt_depth_obs = val

                if rgb_obs is not None and rgb_obs.dtype != np.uint8:
                    rgb_obs = np.clip(rgb_obs * 255, 0, 255).astype(np.uint8)
                if third_rgb is not None and third_rgb.dtype != np.uint8:
                    third_rgb = np.clip(third_rgb * 255, 0, 255).astype(np.uint8)

                step_info = step.get("info", {})
                topdown_map = None
                if "top_down_map" in step_info:
                    td_info = step_info["top_down_map"]
                    target_h = rgb_obs.shape[0] if rgb_obs is not None else 256
                    topdown_map = maps.colorize_draw_agent_and_fit_to_height(
                        td_info, target_h, num_robots=num_robots,
                    )

                frame = compose_unified_frame(
                    rgb_obs, gt_depth_obs, None, topdown_map,
                    third_rgb=third_rgb,
                )

                if save_images and img_dir:
                    Image.fromarray(frame).save(
                        os.path.join(img_dir, f"combined_{i:04d}.png")
                    )

                if save_video:
                    vid_frame = frame
                    max_video_w = 1280
                    if vid_frame.shape[1] > max_video_w:
                        scale = max_video_w / vid_frame.shape[1]
                        new_h = int(vid_frame.shape[0] * scale)
                        new_h = (new_h + 1) & ~1
                        new_w = (max_video_w + 1) & ~1
                        vid_frame = cv2.resize(
                            vid_frame, (new_w, new_h),
                            interpolation=cv2.INTER_AREA,
                        )
                    else:
                        h, w = vid_frame.shape[:2]
                        new_h = (h + 1) & ~1
                        new_w = (w + 1) & ~1
                        if new_h != h or new_w != w:
                            vid_frame = cv2.resize(
                                vid_frame, (new_w, new_h),
                                interpolation=cv2.INTER_AREA,
                            )
                    if writer is None:
                        writer = imageio.get_writer(
                            video_path, fps=fps, quality=5,
                            macro_block_size=1,
                        )
                    writer.append_data(vid_frame)
                    wrote_any = True

            if writer is not None:
                writer.close()
            if not wrote_any and not save_images:
                return ""
        except Exception as e:
            logger.warning(f"Failed to save combined video/images {video_path}: {e}")
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            return ""
        return video_path

    @staticmethod
    def _save_combined_video(
        steps: list, out_dir: str, ep_tag: str,
        metrics: Optional[Dict[str, float]] = None, fps: int = 10,
    ) -> str:
        """Backward-compatible wrapper."""
        return NavThinkerTrainer._save_combined_video_and_images(
            steps, out_dir, ep_tag, metrics=metrics, fps=fps,
        )

    @staticmethod
    def _obs_to_uint8(data, is_depth: bool) -> np.ndarray:
        """Convert an observation tensor/array to a uint8 numpy image."""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        data = data.squeeze()
        if is_depth:
            d_min, d_max = data.min(), data.max()
            if d_max - d_min > 1e-6:
                data = (data - d_min) / (d_max - d_min)
            else:
                data = np.zeros_like(data)
            return (data * 255).astype(np.uint8)
        if data.dtype == np.float32 or data.dtype == np.float64:
            data = np.clip(data * 255, 0, 255).astype(np.uint8)
        return data.astype(np.uint8)

    @staticmethod
    def _save_obs_images(
        steps: list, obs_key: str, img_dir: str, prefix: str,
        is_depth: bool = False,
    ) -> None:
        """Save each frame of ``obs_key`` as PNG.

        Looks in ``step["observations"]`` first, then falls back to
        ``step["raw_visual"]`` for keys stripped from rollout storage.
        """
        try:
            import cv2
            from PIL import Image
            for i, step in enumerate(steps):
                raw = step["observations"].get(obs_key)
                if raw is None:
                    raw = step.get("raw_visual", {}).get(obs_key)
                if raw is None:
                    return
                frame = NavThinkerTrainer._obs_to_uint8(raw, is_depth)
                if is_depth and frame.ndim == 2:
                    bgr = cv2.applyColorMap(frame, cv2.COLORMAP_TURBO)
                    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if frame.ndim == 2:
                    img = Image.fromarray(frame, mode="L")
                else:
                    img = Image.fromarray(frame, mode="RGB")
                img.save(os.path.join(img_dir, f"{prefix}_{i:04d}.png"))
        except Exception as e:
            logger.warning(f"Failed to save {prefix} images to {img_dir}: {e}")

    @staticmethod
    def _save_obs_video(
        steps: list, obs_key: str, out_dir: str, video_name: str,
        is_depth: bool = False, fps: int = 10,
        resize: tuple = None,
    ) -> str:
        """Save an mp4 video from ``obs_key``.

        Looks in ``step["observations"]`` first, then falls back to
        ``step["raw_visual"]`` for keys stripped from rollout storage.

        Args:
            resize: optional (width, height) to resize frames before writing.
        """
        video_path = os.path.join(out_dir, f"{video_name}.mp4")
        try:
            import cv2
            import imageio
            writer = imageio.get_writer(
                video_path, fps=fps, quality=5, macro_block_size=1,
            )
            for step in steps:
                raw = step["observations"].get(obs_key)
                if raw is None:
                    raw = step.get("raw_visual", {}).get(obs_key)
                if raw is None:
                    writer.close()
                    return ""
                frame = NavThinkerTrainer._obs_to_uint8(raw, is_depth)
                if is_depth and frame.ndim == 2:
                    bgr = cv2.applyColorMap(frame, cv2.COLORMAP_TURBO)
                    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                elif frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                if resize is not None:
                    frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
                writer.append_data(frame)
            writer.close()
        except Exception as e:
            logger.warning(f"Failed to save video {video_path}: {e}")
            return ""
        return video_path

    @staticmethod
    def _save_raw_visual_images(
        steps: list, vis_key: str, img_dir: str, prefix: str,
    ) -> None:
        """Save each frame of ``vis_key`` from ``step["raw_visual"]`` as PNG."""
        try:
            from PIL import Image
            for i, step in enumerate(steps):
                raw = step.get("raw_visual", {}).get(vis_key)
                if raw is None:
                    return
                frame = NavThinkerTrainer._obs_to_uint8(raw, is_depth=False)
                img = Image.fromarray(frame, mode="RGB")
                img.save(os.path.join(img_dir, f"{prefix}_{i:04d}.png"))
        except Exception as e:
            logger.warning(f"Failed to save {prefix} images to {img_dir}: {e}")

    @staticmethod
    def _save_raw_visual_video(
        steps: list, vis_key: str, out_dir: str, video_name: str,
        fps: int = 10,
    ) -> str:
        """Save an mp4 video from ``vis_key`` in ``step["raw_visual"]``."""
        video_path = os.path.join(out_dir, f"{video_name}.mp4")
        try:
            import imageio
            writer = imageio.get_writer(
                video_path, fps=fps, quality=5, macro_block_size=1,
            )
            for step in steps:
                raw = step.get("raw_visual", {}).get(vis_key)
                if raw is None:
                    writer.close()
                    return ""
                frame = NavThinkerTrainer._obs_to_uint8(raw, is_depth=False)
                writer.append_data(frame)
            writer.close()
        except Exception as e:
            logger.warning(f"Failed to save video {video_path}: {e}")
            return ""
        return video_path

    @staticmethod
    def _save_topdown_images(steps: list, img_dir: str, num_robots: int = 1) -> None:
        """Save top-down map from ``step["info"]["top_down_map"]`` as PNG."""
        try:
            from PIL import Image
            from habitat.utils.visualizations import maps
            for i, step in enumerate(steps):
                td_info = step.get("info", {}).get("top_down_map")
                if td_info is None:
                    continue
                td_map = maps.colorize_draw_agent_and_fit_to_height(
                    td_info, 256, num_robots=num_robots,
                )
                img = Image.fromarray(td_map)
                img.save(os.path.join(img_dir, f"topdown_{i:04d}.png"))
        except Exception as e:
            logger.warning(f"Failed to save topdown images to {img_dir}: {e}")

    def _save_replay_buffer(self, save_path: str) -> str:
        """Serialize the in-memory replay buffer to a single file.

        In ``collect_replay_only`` mode episodes are already written to disk
        individually by ``_flush_episode_to_disk``, so this method only
        flushes any in-progress (incomplete) episodes and returns.

        In normal (online WM training) mode the full in-memory deque is
        saved as a single snapshot file.

        Returns the path of the saved file (empty string if nothing saved).
        """
        is_collect = getattr(self, "collect_replay_only", False)

        if is_collect:
            n_flushed = 0
            ep_buffers = getattr(self, "_episode_steps", None)
            if ep_buffers:
                for env_idx, buf in enumerate(ep_buffers):
                    if buf:
                        self._flush_episode_to_disk(list(buf), env_idx)
                        buf.clear()
                        n_flushed += 1
            ep_dir = self._get_episode_dir()
            if rank0_only():
                total = len([
                    f for f in os.listdir(ep_dir)
                    if f.startswith("ep_") and f.endswith(".pt")
                ]) if os.path.isdir(ep_dir) else 0
                logger.info(
                    f"Collect-mode flush: {n_flushed} partial episodes flushed, "
                    f"{total} total episode files in {ep_dir}"
                )
            return ep_dir

        # --- Normal (online) mode: single-file snapshot ---
        if self.replay_buffer is None or self._get_replay_buffer_size() == 0:
            return ""

        if not save_path.endswith(".pt"):
            save_path = os.path.join(save_path, "wm_replay_buffer.pt")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        new_data = []
        for env_idx, env_deque in enumerate(self.replay_buffer):
            for exp in env_deque:
                new_data.append((env_idx, exp))

        if not new_data:
            return ""

        obs_space_meta = {}
        sample_obs = new_data[0][1]["observations"]
        for k, v in sample_obs.items():
            obs_space_meta[k] = {"shape": tuple(v.shape), "dtype": str(v.dtype)}
        action_meta = {
            "shape": tuple(new_data[0][1]["actions"].shape),
            "dtype": str(new_data[0][1]["actions"].dtype),
        }

        try:
            torch.save(
                {
                    "experiences": new_data,
                    "num_envs": self.replay_buffer_num_envs,
                    "per_env_capacity": self.replay_buffer_env_capacity,
                    "obs_space_meta": obs_space_meta,
                    "action_meta": action_meta,
                },
                save_path,
            )
            logger.info(
                f"Saved WM replay buffer: {len(new_data)} experiences -> {save_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save replay buffer {save_path}: {e}")
            return ""
        return save_path

    def _load_replay_buffer(self, load_path: str) -> bool:
        """Load replay data into memory for WM training.

        Supports three on-disk layouts (checked in order):

        1. **Episode directory** – a folder of ``ep_NNNNNN.pt`` files
           (produced by collect-only mode).  Each file is one episode.
        2. **Single .pt file** – a monolithic snapshot produced by the
           normal online-training save path.
        3. **Legacy shard files** – ``<base>.shard_NNNN.pt`` next to the
           given path (backward compat).

        Returns True if data was loaded successfully.
        """
        import glob as _glob

        ep_dir = load_path
        if ep_dir.endswith(".pt"):
            ep_dir = os.path.splitext(ep_dir)[0]

        ep_files = sorted(_glob.glob(os.path.join(ep_dir, "ep_*.pt"))) if os.path.isdir(ep_dir) else []

        if ep_files:
            return self._load_episode_files(ep_files)

        # Fallback: single file or legacy shards
        if not load_path.endswith(".pt"):
            load_path = os.path.join(load_path, "wm_replay_buffer.pt")

        base, ext = os.path.splitext(load_path)
        shard_files = sorted(_glob.glob(f"{base}.shard_*{ext}"))
        files_to_load = shard_files if shard_files else ([load_path] if os.path.isfile(load_path) else [])

        if not files_to_load:
            return False

        num_envs = None
        per_env_capacity = None
        all_experiences: list = []

        for fpath in files_to_load:
            data = torch.load(fpath, map_location="cpu", weights_only=False)
            if num_envs is None:
                num_envs = data["num_envs"]
                per_env_capacity = data["per_env_capacity"]
            all_experiences.extend(data["experiences"])

        if num_envs is None:
            return False

        total_exps = len(all_experiences)
        adjusted_capacity = max(per_env_capacity, (total_exps // num_envs) + 1)

        self.replay_buffer = [deque(maxlen=adjusted_capacity) for _ in range(num_envs)]
        self.replay_buffer_num_envs = num_envs
        self.replay_buffer_env_capacity = adjusted_capacity

        for env_idx, exp in all_experiences:
            if env_idx < num_envs:
                self.replay_buffer[env_idx].append(exp)

        last_data = torch.load(files_to_load[-1], map_location="cpu", weights_only=False)
        self._loaded_obs_space_meta = last_data.get("obs_space_meta", {})
        self._loaded_action_meta = last_data.get("action_meta", {})

        total = self._get_replay_buffer_size()
        logger.info(
            f"Loaded WM replay buffer: {total} experiences from "
            f"{len(files_to_load)} file(s) (num_envs={num_envs}, "
            f"per_env_capacity={adjusted_capacity})"
        )
        return True

    def _load_episode_files(self, ep_files: list) -> bool:
        """Index per-episode .pt files for lazy loading.

        Instead of loading all episode data into memory, this method only
        reads the metadata (``num_steps``, ``obs_space_meta``, etc.) from
        each file and builds an index.  Actual episode data is loaded
        on-demand by ``_sample_wm_batch`` with an LRU cache.

        After this call the following attributes are set:

        * ``self._ep_file_index`` – list of ``(path, num_steps)`` for
          episodes long enough to sample from.
        * ``self._ep_cache`` – ``OrderedDict`` used as an LRU cache.
        * ``self._ep_cache_max`` – max number of episodes to keep in cache.
        * ``self._loaded_obs_space_meta`` / ``self._loaded_action_meta``
        * ``self._ep_total_steps`` – total steps across all indexed episodes.
        """
        from collections import OrderedDict

        if not ep_files:
            return False

        first = torch.load(ep_files[0], map_location="cpu", weights_only=False)
        self._loaded_obs_space_meta = first.get("obs_space_meta", {})
        self._loaded_action_meta = first.get("action_meta", {})

        index: list = []
        total_steps = 0
        skipped = 0
        for fpath in ep_files:
            try:
                meta = torch.load(fpath, map_location="cpu", weights_only=False)
                n = meta.get("num_steps", 0)
                if n > 0:
                    index.append((fpath, n))
                    total_steps += n
                else:
                    skipped += 1
            except Exception as e:
                logger.warning(f"Skipping unreadable episode file {fpath}: {e}")
                skipped += 1

        if not index:
            return False

        self._ep_file_index = index
        self._ep_total_steps = total_steps
        self._ep_cache: OrderedDict = OrderedDict()
        self._ep_cache_max = min(64, len(index))

        self.replay_buffer = None
        self.replay_buffer_num_envs = 0
        self.replay_buffer_env_capacity = 0

        lengths = np.array([n for _, n in index])
        logger.info(
            f"Indexed {len(index)} episode files ({total_steps} total steps, "
            f"{skipped} skipped), cache size={self._ep_cache_max}"
        )
        logger.info(
            f"Episode length stats: "
            f"min={lengths.min()}, max={lengths.max()}, "
            f"mean={lengths.mean():.1f}, median={np.median(lengths):.1f}, "
            f"p25={np.percentile(lengths, 25):.0f}, p75={np.percentile(lengths, 75):.0f}"
        )
        seq_len = getattr(self, "wm_sequence_length", 0)
        if seq_len > 0:
            usable = int((lengths >= seq_len).sum())
            logger.info(
                f"With wm_sequence_length={seq_len}: "
                f"{usable}/{len(index)} episodes usable ({100*usable/len(index):.1f}%)"
            )
            if usable < len(index) * 0.5:
                logger.warning(
                    f"Over half of episodes are shorter than wm_sequence_length={seq_len}. "
                    f"Consider reducing wm_sequence_length to {int(np.percentile(lengths, 25))} "
                    f"(p25) or {int(np.median(lengths))} (median)."
                )
        return True

    def _load_episode_from_cache(self, ep_path: str) -> list:
        """Load an episode's steps, using an LRU cache to avoid repeated IO."""
        cache = self._ep_cache
        if ep_path in cache:
            cache.move_to_end(ep_path)
            return cache[ep_path]

        data = torch.load(ep_path, map_location="cpu", weights_only=False)
        steps = data["steps"]
        cache[ep_path] = steps
        if len(cache) > self._ep_cache_max:
            cache.popitem(last=False)
        return steps

    def _store_rollout_to_buffer(self, rollouts):
        """将 rollout 数据存储到 replay buffer。兼容单智能体 RolloutStorage 与多智能体 MultiStorage。

        In ``collect_replay_only`` mode, completed episodes are written to
        disk immediately as individual ``.pt`` files (one per episode) instead
        of being kept in the in-memory deque.  This avoids large memory
        footprints and the need for periodic bulk saves.
        """
        if not self.train_world_model and not getattr(self, "collect_replay_only", False):
            return

        is_multi = (
            hasattr(rollouts, "_active_storages")
            and len(rollouts._active_storages) > 0
        )
        if is_multi:
            all_storages = rollouts._active_storages
            storage = all_storages[0]
        else:
            all_storages = None
            storage = rollouts

        num_steps = getattr(storage, "num_steps", None)
        if num_steps is None:
            return
        buffers = getattr(storage, "buffers", None)
        if buffers is None:
            return

        obs_buf = buffers["observations"]
        actions_buf = buffers["actions"]
        rewards_buf = buffers["rewards"]
        masks_buf = buffers["masks"]
        num_envs = actions_buf.size(1)

        _wm_cfg = self.config.habitat_baselines.get("world_model", None)
        _save_all = _wm_cfg is not None and getattr(_wm_cfg, "save_all_robots", False)
        _num_storages = len(all_storages) if (is_multi and _save_all) else 1

        is_collect = getattr(self, "collect_replay_only", False)

        if is_collect:
            if not hasattr(self, "_episode_steps") or self._episode_steps is None:
                self._episode_steps = [[] for _ in range(num_envs)]
                self._episode_write_counter = self._episode_write_counter if hasattr(self, "_episode_write_counter") else 0
                self._collect_episodes_written = getattr(self, "_collect_episodes_written", 0)
                self._collect_episodes_discarded = getattr(self, "_collect_episodes_discarded", 0)

            step_cache = getattr(self, "_collect_step_cache", {})
            _exhausted = getattr(self, "_envs_exhausted", set())

            for step in range(num_steps):
                for env_idx in range(num_envs):
                    if env_idx in _exhausted:
                        continue

                    obs_dict = {}
                    action_parts = []
                    for si in range(_num_storages):
                        if is_multi and _save_all:
                            _st = all_storages[si]
                            _prefix = f"agent_{si}_"
                        else:
                            _st = storage
                            _prefix = ""
                        _bufs = _st.buffers
                        for key in _bufs["observations"].keys():
                            obs_dict[_prefix + key] = (
                                _bufs["observations"][key][step, env_idx]
                                .detach().clone().cpu()
                            )
                        action_parts.append(
                            _bufs["actions"][step, env_idx].detach().clone().cpu()
                        )

                    if len(action_parts) > 1:
                        merged_actions = torch.cat(action_parts, dim=0)
                    else:
                        merged_actions = action_parts[0]

                    mask = masks_buf[step, env_idx].detach().clone().cpu()

                    raw_visual = {}
                    step_info = {}
                    if env_idx in step_cache and len(step_cache[env_idx]) > 0:
                        cached = step_cache[env_idx].pop(0)
                        raw_visual = cached.get("observations", {})
                        step_info = cached.get("info", {})

                    experience = {
                        "observations": obs_dict,
                        "raw_visual": raw_visual,
                        "info": step_info,
                        "actions": merged_actions,
                        "rewards": rewards_buf[step, env_idx].detach().clone().cpu(),
                        "masks": mask,
                        "is_first": (~mask.bool()).float().cpu(),
                    }
                    self._episode_steps[env_idx].append(experience)

                    _mask_val = mask.item()
                    if not _mask_val and len(self._episode_steps[env_idx]) > 1:
                        ep_metrics = None
                        if hasattr(self, "_episode_done_metrics"):
                            ep_metrics = self._episode_done_metrics.pop(env_idx, None)
                        is_success = (
                            ep_metrics is not None
                            and ep_metrics.get("success", 0) >= 1.0
                            and ep_metrics.get("human_collision", 0) < 1.0
                        )
                        _save_failed = getattr(self, "save_failed_episodes", False)
                        if is_success or _save_failed:
                            saved_path = self._flush_episode_to_disk(
                                self._episode_steps[env_idx], env_idx,
                                metrics=ep_metrics,
                            )
                            if is_success:
                                self._collect_episodes_written += 1
                            else:
                                self._collect_episodes_discarded += 1
                            if not saved_path and rank0_only():
                                scene_id = ep_metrics.get("scene_id", "?") if ep_metrics else "?"
                                ep_id = ep_metrics.get("episode_id", "?") if ep_metrics else "?"
                                logger.info(
                                    f"[Collect] Episode env={env_idx} "
                                    f"scene={scene_id} ep={ep_id} skipped (duplicate or error)"
                                )
                        else:
                            self._collect_episodes_discarded += 1
                        self._episode_steps[env_idx] = []

                        if rank0_only() and (
                            self._collect_episodes_written + self._collect_episodes_discarded
                        ) % 10 == 0:
                            logger.info(
                                f"[Collect] Progress: "
                                f"saved={self._collect_episodes_written}, "
                                f"discarded={self._collect_episodes_discarded}, "
                                f"total={self._collect_episodes_written + self._collect_episodes_discarded}"
                            )
            return

        # --- Normal (online WM training) mode: store in memory deque ---
        if self.replay_buffer is None or self.replay_buffer_num_envs != num_envs:
            per_env_capacity = max(1, self.replay_buffer_size // num_envs)
            self.replay_buffer = [deque(maxlen=per_env_capacity) for _ in range(num_envs)]
            self.replay_buffer_num_envs = num_envs
            self.replay_buffer_env_capacity = per_env_capacity
            if rank0_only():
                logger.info(
                    "Initialized env-wise WM replay buffer: "
                    f"num_envs={num_envs}, per_env_capacity={per_env_capacity}, "
                    f"total_capacity~={per_env_capacity * num_envs}"
                )

        for step in range(num_steps):
            for env_idx in range(num_envs):
                obs_dict = {}
                action_parts = []
                for si in range(_num_storages):
                    if is_multi and _save_all:
                        _st = all_storages[si]
                        _prefix = f"agent_{si}_"
                    else:
                        _st = storage
                        _prefix = ""
                    _bufs = _st.buffers
                    for key in _bufs["observations"].keys():
                        obs_dict[_prefix + key] = (
                            _bufs["observations"][key][step, env_idx]
                            .detach().clone().cpu()
                        )
                    action_parts.append(
                        _bufs["actions"][step, env_idx].detach().clone().cpu()
                    )
                if len(action_parts) > 1:
                    merged_actions = torch.cat(action_parts, dim=0)
                else:
                    merged_actions = action_parts[0]
                mask = masks_buf[step, env_idx].detach().clone().cpu()
                experience = {
                    "observations": obs_dict,
                    "actions": merged_actions,
                    "rewards": rewards_buf[step, env_idx].detach().clone().cpu(),
                    "masks": mask,
                    "is_first": (~mask.bool()).float().cpu(),
                }
                self.replay_buffer[env_idx].append(experience)

    def _sample_wm_batch(self):
        """Sample a batch of contiguous sub-sequences for WM training.

        Two code paths:

        1. **Lazy / episode-file mode** (``self._ep_file_index`` is set) –
           randomly pick episode files weighted by the number of valid
           start positions, load them via the LRU cache, and slice.
        2. **In-memory deque mode** (``self.replay_buffer`` is set) –
           same logic but reads directly from the deque (used during
           online training).
        """
        ep_index = getattr(self, "_ep_file_index", None)

        if ep_index is not None:
            return self._sample_wm_batch_lazy()

        if self.replay_buffer is None:
            return None
        return self._sample_wm_batch_memory()

    def _sample_wm_batch_lazy(self):
        """Sample from episode files on disk (lazy loading with LRU cache)."""
        seq_len = self.wm_sequence_length
        valid_indices = []
        weights = []
        for i, (fpath, n_steps) in enumerate(self._ep_file_index):
            num_starts = n_steps - seq_len + 1
            if num_starts > 0:
                valid_indices.append(i)
                weights.append(num_starts)

        if not valid_indices:
            return None

        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()

        batch_sequences = []
        for _ in range(self.wm_batch_size):
            idx = int(np.random.choice(valid_indices, p=weights))
            fpath, n_steps = self._ep_file_index[idx]
            steps = self._load_episode_from_cache(fpath)
            max_start = len(steps) - seq_len
            start = int(np.random.randint(0, max_start + 1))
            batch_sequences.append(steps[start : start + seq_len])

        return self._collate_sequences(batch_sequences)

    def _sample_wm_batch_memory(self):
        """Sample from in-memory replay buffer (online training)."""
        seq_len = self.wm_sequence_length
        valid_env_ids = []
        env_weights = []
        for env_idx, env_buffer in enumerate(self.replay_buffer):
            num_starts = len(env_buffer) - seq_len + 1
            if num_starts > 0:
                valid_env_ids.append(env_idx)
                env_weights.append(num_starts)

        if not valid_env_ids:
            return None

        env_weights = np.asarray(env_weights, dtype=np.float64)
        env_weights /= env_weights.sum()

        batch_sequences = []
        for _ in range(self.wm_batch_size):
            env_idx = int(np.random.choice(valid_env_ids, p=env_weights))
            env_traj = list(self.replay_buffer[env_idx])
            max_start = len(env_traj) - seq_len
            start = int(np.random.randint(0, max_start + 1))
            batch_sequences.append(env_traj[start : start + seq_len])

        return self._collate_sequences(batch_sequences)

    def _collate_sequences(self, batch_sequences: list) -> dict:
        """Stack a list of step-sequences into a batched tensor dict."""
        batch: dict = {}
        obs_keys = batch_sequences[0][0]["observations"].keys()
        batch["observations"] = {}
        for key in obs_keys:
            batch["observations"][key] = torch.stack([
                torch.stack([step["observations"][key] for step in seq])
                for seq in batch_sequences
            ]).to(self.device)

        for data_key in ["actions", "rewards", "masks", "is_first"]:
            batch[data_key] = torch.stack([
                torch.stack([step[data_key] for step in seq])
                for seq in batch_sequences
            ]).to(self.device)

        return batch

    def _get_wm_sampling_stats(self):
        """Return WM sampling stats: valid episode/env count and avg length."""
        ep_index = getattr(self, "_ep_file_index", None)
        if ep_index is not None:
            seq_len = self.wm_sequence_length
            valid_lengths = [n for _, n in ep_index if n >= seq_len]
            if not valid_lengths:
                return 0, 0.0
            return len(valid_lengths), float(np.mean(valid_lengths))

        if self.replay_buffer is None:
            return 0, 0.0

        valid_env_lengths = [
            len(env_buffer)
            for env_buffer in self.replay_buffer
            if len(env_buffer) >= self.wm_sequence_length
        ]
        if len(valid_env_lengths) == 0:
            return 0, 0.0

        return len(valid_env_lengths), float(np.mean(valid_env_lengths))

    def _compute_kl_loss(self, post, prior):
        """计算 KL divergence with free bits"""
        # KL divergence between posterior and prior
        kl_loss = torch.distributions.kl.kl_divergence(
            self.world_model.dynamics.get_dist(post),
            self.world_model.dynamics.get_dist(prior)
        )

        # Apply free bits (避免 KL collapse)
        # Free bits: 只惩罚 KL > kl_free_bits 的部分
        kl_loss = torch.clamp(kl_loss - self.kl_free_bits, min=0.0)

        return kl_loss.mean()

    def _update_world_model(self, update):
        """训练 World Model"""
        self._set_world_model_grad_enabled(True)
        self.wm_train_model.train()

        if rank0_only():
            logger.info(f"[Update {update}] Updating World Model...")
            valid_env_count, avg_env_length = self._get_wm_sampling_stats()
            logger.info(
                f"[Update {update}] WM sampling stats: "
                f"valid_envs={valid_env_count}, avg_env_length={avg_env_length:.1f}"
            )

        wm_losses_sum = {
            'depth_loss': 0.0,
            'depth_pred_loss': 0.0,
            'traj_loss': 0.0,
            'reward_loss': 0.0,
            'kl_loss': 0.0,
            'feature_loss': 0.0,
            'imag_depth_loss': 0.0,
            'imag_reward_loss': 0.0,
            'imag_traj_loss': 0.0,
            'total_loss': 0.0,
        }

        num_batches_trained = 0
        last_batch = None

        # 训练多个 epochs
        for epoch in range(self.wm_epochs_per_update):
            # 1. 从 replay buffer 采样 sequences
            batch = self._sample_wm_batch()

            # DDP safety: all ranks must make the same control-flow decision.
            if self._is_distributed and torch.distributed.is_initialized():
                has_batch = torch.tensor(
                    1 if batch is not None else 0,
                    device=self.device,
                    dtype=torch.int32,
                )
                torch.distributed.all_reduce(
                    has_batch, op=torch.distributed.ReduceOp.MIN
                )
                global_has_batch = int(has_batch.item()) == 1
                if not global_has_batch:
                    if rank0_only():
                        logger.warning(
                            f"[Update {update}] Global replay buffer too small, "
                            "skipping WM update this round"
                        )
                    break

            if batch is None:
                if rank0_only():
                    logger.warning(f"[Update {update}] Replay buffer too small, skipping WM update")
                break

            # WMP-style alignment: always treat the first step of each sampled
            # sequence as a new sequence start.
            batch['is_first'][:, 0] = 1.0

            # 2. WM forward pass (through DDP wrapper if enabled)
            wm_losses = self.wm_train_model(
                batch, self.kl_free_bits,
                kl_dyn_scale=self.kl_dyn_scale,
                kl_rep_scale=self.kl_rep_scale,
                imag_loss_scale=self.imag_loss_scale,
                imag_loss_depth_scale=self.imag_loss_depth_scale,
                imag_loss_reward_scale=self.imag_loss_reward_scale,
                imag_loss_traj_scale=self.imag_loss_traj_scale,
                imag_loss_horizon_gamma=self.imag_loss_horizon_gamma,
                imag_loss_horizon=self.imag_loss_horizon,
                feature_loss_scale=self.feature_loss_scale,
                depth_pred_loss_scale=self.depth_pred_loss_scale,
            )
            depth_loss = wm_losses["depth_loss"]
            depth_pred_loss = wm_losses["depth_pred_loss"]
            traj_loss = wm_losses["traj_loss"]
            reward_loss = wm_losses["reward_loss"]
            kl_loss = wm_losses["kl_loss"]
            feature_loss = wm_losses["feature_loss"]
            imag_depth_loss = wm_losses["imag_depth_loss"]
            imag_reward_loss = wm_losses["imag_reward_loss"]
            imag_traj_loss = wm_losses["imag_traj_loss"]

            # 4. Total WM loss (reconstruction + KL + feature + imagination)
            wm_loss = (
                depth_loss * self.depth_loss_scale +
                depth_pred_loss * self.depth_pred_loss_scale +
                traj_loss * self.traj_loss_scale +
                reward_loss * self.reward_loss_scale +
                kl_loss * self.kl_loss_scale +
                feature_loss * self.feature_loss_scale
            )
            if self.imag_loss_scale > 0:
                wm_loss = wm_loss + self.imag_loss_scale * (
                    imag_depth_loss * self.imag_loss_depth_scale +
                    imag_reward_loss * self.imag_loss_reward_scale +
                    imag_traj_loss * self.imag_loss_traj_scale
                )

            # 5. Backward and optimizer step
            self.wm_optimizer.zero_grad()
            wm_loss.backward()

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.world_model.parameters(),
                self.wm_grad_clip
            )

            self.wm_optimizer.step()

            # 累积 losses
            wm_losses_sum['depth_loss'] += depth_loss.item()
            dp_l = depth_pred_loss.item() if torch.is_tensor(depth_pred_loss) else depth_pred_loss
            wm_losses_sum['depth_pred_loss'] += dp_l
            wm_losses_sum['traj_loss'] += traj_loss.item()
            wm_losses_sum['reward_loss'] += reward_loss.item()
            wm_losses_sum['kl_loss'] += kl_loss.item()
            feat_l = feature_loss.item() if torch.is_tensor(feature_loss) else feature_loss
            wm_losses_sum['feature_loss'] = wm_losses_sum.get('feature_loss', 0.0) + feat_l
            imag_d = imag_depth_loss.item() if torch.is_tensor(imag_depth_loss) else imag_depth_loss
            imag_r = imag_reward_loss.item() if torch.is_tensor(imag_reward_loss) else imag_reward_loss
            imag_t = imag_traj_loss.item() if torch.is_tensor(imag_traj_loss) else imag_traj_loss
            wm_losses_sum['imag_depth_loss'] += imag_d
            wm_losses_sum['imag_reward_loss'] += imag_r
            wm_losses_sum['imag_traj_loss'] += imag_t
            wm_losses_sum['total_loss'] += wm_loss.item()
            num_batches_trained += 1
            last_batch = batch

        # Keep WM optimizer path isolated from PPO path.
        self._clear_world_model_grads()
        self._set_world_model_grad_enabled(False)

        # 平均 losses
        if num_batches_trained > 0:
            for key in wm_losses_sum:
                wm_losses_sum[key] /= num_batches_trained

        if rank0_only():
            msg = (
                f"[Update {update}] WM training completed: "
                f"depth={wm_losses_sum['depth_loss']:.3f}, "
                f"traj={wm_losses_sum['traj_loss']:.3f}, "
                f"reward={wm_losses_sum['reward_loss']:.3f}, "
                f"kl={wm_losses_sum['kl_loss']:.3f}, "
                f"total={wm_losses_sum['total_loss']:.3f}"
            )
            if self.depth_pred_loss_scale > 0:
                msg += f", depth_pred={wm_losses_sum.get('depth_pred_loss', 0.0):.3f}"
            if self.feature_loss_scale > 0:
                msg += f", feat={wm_losses_sum.get('feature_loss', 0.0):.3f}"
            if self.imag_loss_scale > 0:
                msg += (
                    f" | imag: depth={wm_losses_sum['imag_depth_loss']:.3f}, "
                    f"reward={wm_losses_sum['imag_reward_loss']:.3f}, "
                    f"traj={wm_losses_sum['imag_traj_loss']:.3f}"
                )
            logger.info(msg)

        # Visualize a few samples from the last batch
        if rank0_only() and last_batch is not None:
            self._last_wm_vis_batch = last_batch

        return wm_losses_sum

    # ------------------------------------------------------------------
    # Social Imagination
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_social_imagination(self, rollouts):
        """Inject social safety reward by predicting human proximity from WM.

        When ``imagine_horizon == 1`` **and** lookahead features are cached in
        the rollout storage, we reuse them directly (zero Transformer forward
        passes).  The cached ``lookahead_features`` tensor has shape
        ``(T, B, D*A)`` where A is the number of discrete actions and D is the
        pooled WM feature dim.  We index into it with the actually-chosen
        action to obtain the per-step imagination feature, then run only the
        lightweight ``human_traj`` head to compute the social penalty.

        For ``imagine_horizon > 1`` or when lookahead features are unavailable,
        falls back to the full imagination rollout path.
        """
        if not self.imagine_enabled:
            return {}

        wm = self.world_model
        if wm is None:
            return {}
        wm.eval()

        storage = self._get_agent0_storage(rollouts)
        num_steps = storage.buffers["rewards"].shape[0]
        num_envs = storage.buffers["rewards"].shape[1]
        H = self.imagine_horizon

        # === Fast path: reuse rollout-cached lookahead features (H==1) ===
        la_buf = storage.buffers.get("lookahead_features")
        if H == 1 and la_buf is not None:
            return self._social_imagination_from_cache(
                wm, storage, la_buf, num_steps, num_envs,
            )

        # === Fallback: full imagination rollout (H > 1 or no cache) ===
        if not hasattr(self, '_wm_last_post_states') or self._wm_last_post_states is None:
            return {}

        post_states = self._wm_last_post_states
        human_sg = self._wm_last_human_state_goal
        num_actions = wm.dynamics._num_actions

        all_actions = storage.buffers["actions"][:num_steps].squeeze(-1).long()
        all_actions_oh = F.one_hot(all_actions, num_actions).float()
        max_start = min(num_steps, post_states["deter"].shape[0])

        total_social_reward = 0.0
        total_intrinsic_reward = 0.0
        steps_augmented = 0

        for step_idx in range(max_start):
            remaining = num_steps - step_idx
            h = min(H, remaining)
            if h < 1:
                continue

            start_state = {k: v[step_idx] for k, v in post_states.items()}
            hsg = human_sg[step_idx] if human_sg is not None else None

            future_actions = all_actions_oh[step_idx:step_idx + h]
            state = {k: v.detach() for k, v in start_state.items()}
            imag_rewards_list = []
            imag_trajs_list = []

            for t in range(h):
                action = future_actions[t]
                state = wm.dynamics.img_step(state, action)
                feat = wm.dynamics.get_feat(state)
                r = wm.heads["reward"](feat).mean()
                tr = wm.heads["human_traj"](
                    feat, human_state_goal=hsg
                ).mean()
                imag_rewards_list.append(r)
                imag_trajs_list.append(tr)

            imag_rewards = torch.stack(imag_rewards_list, dim=1)
            imag_trajs = torch.stack(imag_trajs_list, dim=1)

            last_human_pos = imag_trajs[:, :, :, -1, :]
            human_dists = last_human_pos.norm(dim=-1)
            min_dist = human_dists.min(dim=-1).values
            violation = torch.clamp(
                self.imagine_safety_margin - min_dist, min=0.0
            )
            social_penalty = -violation.mean(dim=-1)

            if h > 1:
                intrinsic = imag_rewards.squeeze(-1).std(dim=-1)
            else:
                intrinsic = torch.zeros(num_envs, device=self.device)

            combined = (
                self.imagine_social_coeff * social_penalty
                + self.imagine_intrinsic_coeff * intrinsic
            )

            storage.buffers["rewards"][step_idx] += combined.unsqueeze(-1)

            total_social_reward += social_penalty.mean().item()
            total_intrinsic_reward += intrinsic.mean().item()
            steps_augmented += 1

        stats = {}
        if steps_augmented > 0:
            stats["imagine_social_reward"] = total_social_reward / steps_augmented
            stats["imagine_intrinsic_reward"] = total_intrinsic_reward / steps_augmented
            stats["imagine_steps"] = steps_augmented
            if rank0_only():
                logger.info(
                    f"[Imagination] Augmented {steps_augmented} steps (H={H}): "
                    f"social={stats['imagine_social_reward']:.4f}, "
                    f"intrinsic={stats['imagine_intrinsic_reward']:.4f}"
                )
        return stats

    @torch.no_grad()
    def _social_imagination_from_cache(
        self, wm, storage, la_buf, num_steps, num_envs,
    ):
        """Reuse rollout-cached lookahead features for social imagination.

        ``la_buf`` shape: (T+1, B, D*A) — the concatenated pooled features
        from ``_run_lookahead`` for all A actions.  We select the feature
        corresponding to the actually-chosen action, then run only the
        ``human_traj`` decoder head (no Transformer forward pass needed).
        """
        num_actions = wm.dynamics._num_actions
        T = num_steps
        B = num_envs

        # la_buf: (T+1, B, D*A), actions: (T, B, 1)
        la_feats = la_buf[:T]  # (T, B, D*A)
        D = la_feats.shape[-1] // num_actions  # pooled feature dim per action

        actions = storage.buffers["actions"][:T].squeeze(-1).long()  # (T, B)

        # Reshape to (T, B, A, D) and gather the chosen action's feature
        la_per_action = la_feats.view(T, B, num_actions, D)
        action_idx = actions.unsqueeze(-1).unsqueeze(-1).expand(T, B, 1, D)
        chosen_feat = la_per_action.gather(2, action_idx).squeeze(2)  # (T, B, D)

        # Get human_state_goal from observations
        obs = storage.buffers["observations"]
        hsg_key = next((k for k in obs if "human_state_goal" in k), None)
        hsg = obs[hsg_key][:T] if hsg_key is not None else None  # (T, B, ...)

        # Process in chunks of B to keep memory bounded
        all_social = []
        for t in range(T):
            feat_t = chosen_feat[t]  # (B, D)
            hsg_t = hsg[t] if hsg is not None else None

            traj = wm.heads["human_traj"](
                feat_t, human_state_goal=hsg_t
            ).mean()  # (B, N_humans, T_pred, 2)

            last_pos = traj[:, :, -1, :]  # (B, N_humans, 2)
            dists = last_pos.norm(dim=-1)  # (B, N_humans)
            min_d = dists.min(dim=-1).values  # (B,)
            violation = torch.clamp(
                self.imagine_safety_margin - min_d, min=0.0
            )
            all_social.append(-violation)

        social_penalty = torch.stack(all_social, dim=0)  # (T, B)
        combined = self.imagine_social_coeff * social_penalty

        storage.buffers["rewards"][:T] += combined.unsqueeze(-1)

        avg_social = social_penalty.mean().item()
        stats = {
            "imagine_social_reward": avg_social,
            "imagine_intrinsic_reward": 0.0,
            "imagine_steps": T,
        }
        if rank0_only():
            logger.info(
                f"[Imagination] From cache: {T} steps (0 Transformer fwd), "
                f"social={avg_social:.4f}"
            )
        return stats

    def _get_agent0_storage(self, rollouts):
        """Extract agent_0's RolloutStorage from MultiStorage or return as-is."""
        if hasattr(rollouts, 'buffers'):
            return rollouts
        if hasattr(rollouts, '_active_storages') and len(rollouts._active_storages) > 0:
            return rollouts._active_storages[0]
        raise AttributeError(
            f"Cannot extract agent_0 storage from {type(rollouts).__name__}"
        )

    @torch.no_grad()
    def _cache_wm_states_for_imagination(self, rollouts):
        """Cache RSSM posterior states from the current rollout for imagination.

        Called after rollout collection and before PPO update so that
        ``_run_social_imagination`` can use the states.
        """
        if not self.imagine_enabled:
            return
        if self.world_model is None:
            return

        wm = self.world_model
        wm.eval()

        storage = self._get_agent0_storage(rollouts)
        num_steps = storage.buffers["actions"].shape[0]
        num_envs = storage.buffers["actions"].shape[1]

        obs_dict = {}
        for key in storage.buffers["observations"]:
            val = storage.buffers["observations"][key][:num_steps]
            obs_dict[key] = val

        flat_obs = {}
        for key, val in obs_dict.items():
            flat_obs[key] = val.reshape(num_steps * num_envs, *val.shape[2:])

        _is_dino_wm = getattr(wm, '_is_dino_wm', False)
        flat_embed = wm.encoder(flat_obs)

        if _is_dino_wm:
            cached_vit = getattr(wm.encoder, '_cached_vit_patch_features', None)
            embed = cached_vit.reshape(num_steps, num_envs, *cached_vit.shape[1:])
            embed = embed.permute(1, 0, *range(2, embed.dim()))  # (B, T, N, D)
        else:
            embed = flat_embed.reshape(num_steps, num_envs, -1)
            embed = embed.permute(1, 0, 2)  # (B, T, embed)

        actions = storage.buffers["actions"][:num_steps].squeeze(-1).long()
        actions = actions.permute(1, 0)  # (B, T)
        actions_oh = F.one_hot(
            actions, num_classes=wm.dynamics._num_actions,
        ).float()

        is_first = torch.zeros(num_envs, num_steps, device=self.device)
        is_first[:, 0] = 1.0

        post, _ = wm.dynamics.observe(embed, actions_oh, is_first)
        # post keys have shape (B, T, ...) — transpose to (T, B, ...)
        self._wm_last_post_states = {
            k: v.permute(1, 0, *range(2, v.dim())) for k, v in post.items()
        }

        hsg_key = next(
            (k for k in obs_dict if "human_state_goal" in k), None
        )
        if hsg_key is not None:
            self._wm_last_human_state_goal = obs_dict[hsg_key]  # (T, B, N, dim)
        else:
            self._wm_last_human_state_goal = None

    @torch.no_grad()
    def _visualize_wm_predictions(self, writer, step, num_samples=3):
        """Generate WM prediction visualizations (vertical-stack layout).

        Follows the offline inference visualization style:
          Row 0: RGB observation (if available in batch)
          Row 1: History panel (GT depth thumbnails with action icons)
          Row 2: Step label with action info
          Row 3: [GT Depth | DPT(GT Feat) | DPT(Pred Feat)]  with RMSE
          Row 4: [GT Traj | Pred Traj]
          Row 5: [GT ViT PCA | Pred ViT PCA]
          Row 6: [Cosine Similarity Heatmap]
          Row 7: [GT Depth | Probe(GT ViT) | Probe(Pred ViT)]
        """
        import cv2

        batch = getattr(self, '_last_wm_vis_batch', None)
        if batch is None:
            return

        wm = self.world_model
        wm.eval()

        B, T = batch["actions"].shape[:2]
        num_samples = min(num_samples, B)
        wm_impl = getattr(wm, "module", wm)
        is_dino_wm = getattr(wm_impl, "_is_dino_wm", False)

        flat_obs = {}
        for key, val in batch["observations"].items():
            flat_obs[key] = val.reshape(B * T, *val.shape[2:])

        flat_embed = wm.encoder(flat_obs)
        cached_vit_features = getattr(wm.encoder, '_cached_vit_patch_features', None)

        if is_dino_wm:
            embed = cached_vit_features.reshape(B, T, *cached_vit_features.shape[1:])
        else:
            embed = flat_embed.reshape(B, T, -1)

        actions_oh = F.one_hot(
            batch["actions"].long().squeeze(-1),
            num_classes=wm.dynamics._num_actions,
        ).float()

        post, _ = wm.dynamics.observe(embed, actions_oh, batch["is_first"])
        feat = wm.dynamics.get_feat(post)

        pred_patch_feat = None
        if is_dino_wm:
            pred_patch_feat = wm.get_pred_patch_feat(post)
            depth_dist = wm.heads["depth"](pred_patch_feat)
        else:
            depth_dist = wm.heads["depth"](feat)
        pred_depth = depth_dist.mean()

        gt_recon_depth = None
        if is_dino_wm:
            gt_depth_dist = wm.heads["depth"](embed)
            gt_recon_depth = gt_depth_dist.mean()

        pred_vit_features = None
        if is_dino_wm:
            pred_vit_features = pred_patch_feat

        human_state_goal = next(
            (batch["observations"][k] for k in batch["observations"] if "human_state_goal" in k),
            None,
        )

        traj_dist = wm.heads["human_traj"](feat, human_state_goal=human_state_goal)
        pred_traj = traj_dist.mean()

        depth_key = next((k for k in batch["observations"] if "depth" in k.lower()), None)
        rgb_key = next(
            (k for k in batch["observations"]
             if "rgb" in k.lower() and len(batch["observations"][k].shape) >= 4),
            None,
        )
        traj_key = next((k for k in batch["observations"] if "future_trajectory" in k.lower()), None)
        human_num_key = next((k for k in batch["observations"] if "human_num" in k.lower()), None)

        vis_dir = os.path.join(
            self.config.habitat_baselines.checkpoint_folder, "wm_vis"
        )
        os.makedirs(vis_dir, exist_ok=True)

        gt_vit_bt = None
        if cached_vit_features is not None:
            gt_vit_bt = cached_vit_features.reshape(
                B, T, *cached_vit_features.shape[1:]
            )

        num_hist = getattr(wm.dynamics, '_num_hist', 0) if is_dino_wm else 0
        if is_dino_wm:
            mid_t = T - 1
        else:
            mid_t = T // 2

        for i in range(num_samples):
            panels = []

            # RGB observation panel (if available)
            if rgb_key is not None and mid_t > 0:
                hist_end = mid_t if is_dino_wm else mid_t
                hist_start = max(0, hist_end - max(num_hist, 1))
                rgb_frames = []
                rgb_actions = []
                rgb_times = []
                for h_t in range(hist_start, hist_end + 1):
                    h_rgb = batch["observations"][rgb_key][i, h_t].cpu().numpy()
                    rgb_frames.append(h_rgb)
                    h_act = int(batch["actions"][i, h_t].item()) if h_t < T else None
                    rgb_actions.append(h_act)
                    rgb_times.append(h_t)
                if rgb_frames:
                    try:
                        rgb_panel = render_rgb_observation(
                            rgb_frames, rgb_times, rgb_actions,
                        )
                        if rgb_panel is not None:
                            panels.append(rgb_panel)
                    except Exception as e:
                        logger.warning(f"[WM Vis] RGB observation panel failed: {e}")

            # History panel: GT depth thumbnails with action icons
            if depth_key is not None and mid_t > 0:
                hist_end = mid_t if is_dino_wm else mid_t
                hist_start = max(0, hist_end - max(num_hist, 1))
                hist_depths = []
                hist_actions = []
                hist_times = []
                for h_t in range(hist_start, hist_end):
                    h_d = batch["observations"][depth_key][i, h_t].cpu().numpy()
                    if h_d.max() > 1.0:
                        h_d = h_d / 255.0
                    if h_d.ndim == 3:
                        h_d = h_d[..., 0]
                    hist_depths.append(h_d)
                    h_act = int(batch["actions"][i, h_t].item()) if h_t < T else None
                    hist_actions.append(h_act)
                    hist_times.append(h_t)
                if hist_depths:
                    try:
                        hist_panel = render_history_panel(
                            hist_depths, hist_actions, hist_times,
                        )
                        if hist_panel is not None:
                            panels.append(hist_panel)
                    except Exception as e:
                        logger.warning(f"[WM Vis] History panel failed: {e}")

            # Step label for the predicted frame
            pred_action = int(batch["actions"][i, mid_t].item()) if mid_t < T else None
            panels.append(render_step_label(
                f"Prediction at t={mid_t}", width=768, action_id=pred_action,
            ))

            gt_d = None
            if depth_key is not None:
                gt_d = batch["observations"][depth_key][i, mid_t].cpu().numpy()
                if gt_d.max() > 1.0:
                    gt_d = gt_d / 255.0
                if gt_d.ndim == 3:
                    gt_d = gt_d[..., 0]

                pd_d = pred_depth[i, mid_t].cpu().numpy()
                if pd_d.ndim == 3:
                    pd_d = pd_d[..., 0]
                if gt_d.shape != pd_d.shape:
                    pd_d = cv2.resize(pd_d, (gt_d.shape[1], gt_d.shape[0]))

                if gt_recon_depth is not None:
                    dpt_gt_np = gt_recon_depth[i, mid_t].cpu().numpy()
                    if dpt_gt_np.ndim == 3:
                        dpt_gt_np = dpt_gt_np[..., 0]
                    if gt_d.shape != dpt_gt_np.shape:
                        dpt_gt_np = cv2.resize(dpt_gt_np, (gt_d.shape[1], gt_d.shape[0]))
                    try:
                        depth_panel = render_depth_triple(gt_d, dpt_gt_np, pd_d)
                        panels.append(depth_panel)
                    except Exception as e:
                        logger.warning(f"[WM Vis] DPT depth triple failed: {e}")
                else:
                    try:
                        depth_panel = render_depth_comparison(gt_d, pd_d)
                        panels.append(depth_panel)
                    except Exception as e:
                        logger.warning(f"[WM Vis] Depth comparison failed: {e}")

            if traj_key is not None:
                gt_tr = batch["observations"][traj_key][i, mid_t].cpu().numpy()
                pd_tr = pred_traj[i, mid_t].cpu().numpy()
                if human_num_key is not None:
                    nh = int(batch["observations"][human_num_key][i, mid_t].item())
                else:
                    nh = gt_tr.shape[0]
                hp = None
                if human_state_goal is not None:
                    hp = human_state_goal[i, mid_t].cpu().numpy()[:, :2]
                try:
                    traj_panel = render_trajectory_comparison(gt_tr, pd_tr, nh, human_positions=hp)
                    panels.append(traj_panel)
                except Exception as e:
                    logger.warning(f"[WM Vis] Trajectory failed: {e}")

            if pred_vit_features is not None and gt_vit_bt is not None:
                gt_vit_i = gt_vit_bt[i, mid_t].cpu().numpy()
                pd_vit_i = pred_vit_features[i, mid_t].cpu().numpy()
                n_vit = gt_vit_i.shape[0]
                h_vit = int(n_vit ** 0.5)
                w_vit = h_vit if h_vit * h_vit == n_vit else n_vit // h_vit

                try:
                    feat_panel = render_feature_pca_comparison(
                        gt_vit_i, pd_vit_i, h_vit, w_vit,
                    )
                    panels.append(feat_panel)
                except Exception as e:
                    logger.warning(f"[WM Vis] Feature PCA failed: {e}")

                try:
                    cos_panel = render_cosine_sim_heatmap(
                        gt_vit_i, pd_vit_i, h_vit, w_vit,
                    )
                    panels.append(cos_panel)
                except Exception as e:
                    logger.warning(f"[WM Vis] CosSim heatmap failed: {e}")

                if gt_d is not None:
                    try:
                        probe = LinearDepthProbe()
                        probe.fit(gt_vit_i, gt_d)
                        th, tw = gt_d.shape[:2]
                        probe_gt_depth = probe.predict(gt_vit_i, th, tw)
                        probe_pd_depth = probe.predict(pd_vit_i, th, tw)
                        probe_panel = render_probe_depth_comparison(
                            gt_d, probe_gt_depth, probe_pd_depth,
                        )
                        panels.append(probe_panel)
                    except Exception as e:
                        logger.warning(f"[WM Vis] Depth probe failed: {e}")

            frame = vstack_wm_panels(panels, target_w=768)
            if frame is not None:
                save_path = os.path.join(vis_dir, f"step{step:08d}_sample{i}.png")
                cv2.imwrite(save_path, frame)

                if writer is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                    writer.add_image(f"wm_vis/sample_{i}", frame_tensor, step)

        logger.info(f"[WM Vis] Saved {num_samples} visualization(s) to {vis_dir}")
        self._last_wm_vis_batch = None

    # ------------------------------------------------------------------
    # Layer-2 Imagination Verification: imagine vs reality
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _visualize_imagination_vs_reality(self, writer, step, num_samples=3,
                                          horizon=5):
        """Compare imagined future with real future from replay buffer.

        Pipeline:
          1. Sample a batch from replay buffer (length = seq_len).
          2. Observe first half → posterior state at mid_t.
          3. Imagine H steps from mid_t using real actions from the batch.
          4. Compare imagined reward / traj / depth with real observations.
          5. Save multi-panel figure to disk and TensorBoard.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        batch = self._sample_wm_batch()
        if batch is None:
            return

        wm = self.world_model
        wm.eval()

        B, T = batch["actions"].shape[:2]
        num_samples = min(num_samples, B)
        mid_t = T // 2
        H = min(horizon, T - mid_t - 1)
        if H < 2:
            return

        # --- Encode full sequence and observe ---
        flat_obs = {}
        for key, val in batch["observations"].items():
            flat_obs[key] = val.reshape(B * T, *val.shape[2:])

        wm_impl = getattr(wm, "module", wm)
        is_dino_wm = getattr(wm_impl, "_is_dino_wm", False)

        flat_embed = wm.encoder(flat_obs)
        cached_vit_features = getattr(wm.encoder, '_cached_vit_patch_features', None)

        if is_dino_wm:
            embed = cached_vit_features.reshape(B, T, *cached_vit_features.shape[1:])
        else:
            embed = flat_embed.reshape(B, T, -1)

        actions_oh = F.one_hot(
            batch["actions"].long().squeeze(-1),
            num_classes=wm.dynamics._num_actions,
        ).float()

        post, prior = wm.dynamics.observe(embed, actions_oh, batch["is_first"])

        # --- Posterior reconstruction at all timesteps ---
        feat_post = wm.dynamics.get_feat(post)
        post_reward_dist = wm.heads["reward"](feat_post)
        post_rewards = post_reward_dist.mean()  # (B, T, 1)

        hsg_key = next(
            (k for k in batch["observations"] if "human_state_goal" in k), None
        )
        human_state_goal = batch["observations"][hsg_key] if hsg_key else None

        post_traj_dist = wm.heads["human_traj"](
            feat_post, human_state_goal=human_state_goal
        )
        post_trajs = post_traj_dist.mean()  # (B, T, N, T_pred, 2)

        # --- Imagination from mid_t using real actions ---
        start_state = {k: v[:, mid_t] for k, v in post.items()}
        hsg_mid = human_state_goal[:, mid_t] if human_state_goal is not None else None

        real_future_actions = actions_oh[:, mid_t:mid_t + H]  # (B, H, num_actions)

        imag_feats, imag_rewards, imag_trajs, imag_depths = [], [], [], []
        state = {k: v.detach() for k, v in start_state.items()}

        if hasattr(wm.dynamics, 'reset_imagination'):
            wm.dynamics.reset_imagination()
        if hasattr(wm.dynamics, 'reset_kv_cache'):
            wm.dynamics.reset_kv_cache(B)

        for t in range(H):
            action = real_future_actions[:, t]
            state = wm.dynamics.img_step(state, action)
            feat = wm.dynamics.get_feat(state)
            r = wm.heads["reward"](feat).mean()
            tr = wm.heads["human_traj"](feat, human_state_goal=hsg_mid).mean()
            if is_dino_wm:
                pf = wm.get_patch_feat(state)
                d = wm.heads["depth"](pf).mean()
            else:
                d = wm.heads["depth"](feat).mean()
            imag_feats.append(feat)
            imag_rewards.append(r)
            imag_trajs.append(tr)
            imag_depths.append(d)
        imag_depths_no_skip = None

        imag_rewards = torch.stack(imag_rewards, dim=1)  # (B, H, 1)
        imag_trajs = torch.stack(imag_trajs, dim=1)      # (B, H, N, T_pred, 2)
        imag_depths = torch.stack(imag_depths, dim=1)     # (B, H, C, H_img, W_img) or similar

        # --- Also get prior-only predictions for comparison ---
        prior_feat = wm.dynamics.get_feat(prior)
        prior_reward_dist = wm.heads["reward"](prior_feat)
        prior_rewards = prior_reward_dist.mean()  # (B, T, 1)

        # --- KL divergence (skip for deterministic dynamics like DINO-WM) ---
        if is_dino_wm or not hasattr(wm.dynamics, 'get_dist'):
            kl_vals = torch.zeros(B, T, device=feat_post.device)
        else:
            import torch.distributions as torchd
            post_dist = wm.dynamics.get_dist(post)
            prior_dist = wm.dynamics.get_dist(prior)
            kl_vals = torchd.kl.kl_divergence(post_dist, prior_dist)  # (B, T)

        # --- Convert to numpy ---
        real_rewards = batch["rewards"].cpu().numpy()       # (B, T, 1)
        post_rewards_np = post_rewards.cpu().numpy()        # (B, T, 1)
        prior_rewards_np = prior_rewards.cpu().numpy()      # (B, T, 1)
        imag_rewards_np = imag_rewards.cpu().numpy()        # (B, H, 1)
        kl_np = kl_vals.cpu().numpy()                       # (B, T)

        traj_key = next(
            (k for k in batch["observations"] if "future_trajectory" in k.lower()),
            None,
        )
        real_trajs = batch["observations"][traj_key].cpu().numpy() if traj_key else None
        post_trajs_np = post_trajs.cpu().numpy()
        imag_trajs_np = imag_trajs.cpu().numpy()

        hn_key = next(
            (k for k in batch["observations"] if "human_num" in k.lower()), None
        )

        # --- Depth: posterior reconstruction + imagined ---
        depth_key = next(
            (k for k in batch["observations"] if "depth" in k.lower()), None
        )
        if is_dino_wm:
            patch_feat_post = wm.get_patch_feat(post)
            post_depth_dist = wm.heads["depth"](patch_feat_post)
        else:
            post_depth_dist = wm.heads["depth"](feat_post)
        post_depths_np = post_depth_dist.mean().cpu().numpy()  # (B, T, ...)
        imag_depths_np = imag_depths.cpu().numpy()             # (B, H, ...)
        imag_depths_no_skip_np = (
            imag_depths_no_skip.cpu().numpy() if imag_depths_no_skip is not None else None
        )

        # --- Output directory ---
        vis_dir = os.path.join(
            self.config.habitat_baselines.checkpoint_folder, "imag_vs_real"
        )
        os.makedirs(vis_dir, exist_ok=True)

        # --- Generate per-sample figure ---
        for i in range(num_samples):
            fig = plt.figure(figsize=(20, 16))
            gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

            # ---- Panel 1: Reward comparison ----
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(range(T), real_rewards[i, :, 0], "b-", alpha=0.7,
                     label="Real")
            ax1.plot(range(T), post_rewards_np[i, :, 0], "g--", alpha=0.7,
                     label="Posterior")
            ax1.plot(range(T), prior_rewards_np[i, :, 0], "c:", alpha=0.5,
                     label="Prior (observe)")
            imag_x = range(mid_t + 1, mid_t + 1 + H)
            ax1.plot(imag_x, imag_rewards_np[i, :, 0], "r-", linewidth=2,
                     label="Imagined (prior)")
            ax1.axvline(x=mid_t, color="k", linestyle=":", alpha=0.5)
            ax1.set_xlabel("Timestep")
            ax1.set_ylabel("Reward")
            ax1.set_title("Reward: Real vs Posterior vs Imagined")
            ax1.legend(fontsize=7)
            ax1.grid(True, alpha=0.3)

            # ---- Panel 2: Reward error (imagined vs real) ----
            ax2 = fig.add_subplot(gs[0, 1])
            real_future_r = real_rewards[i, mid_t + 1:mid_t + 1 + H, 0]
            imag_r = imag_rewards_np[i, :, 0]
            post_future_r = post_rewards_np[i, mid_t + 1:mid_t + 1 + H, 0]
            min_len = min(len(real_future_r), len(imag_r))
            if min_len > 0:
                ax2.plot(range(min_len), np.abs(imag_r[:min_len] - real_future_r[:min_len]),
                         "r-o", markersize=3, label="|Imag - Real|")
                ax2.plot(range(min_len), np.abs(post_future_r[:min_len] - real_future_r[:min_len]),
                         "g--s", markersize=3, label="|Post - Real|")
                ax2.set_xlabel("Steps after mid_t")
                ax2.set_ylabel("Absolute Error")
                ax2.set_title("Reward Prediction Error")
                ax2.legend(fontsize=7)
                ax2.grid(True, alpha=0.3)

            # ---- Panel 3: KL divergence ----
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.plot(range(T), kl_np[i], "m-", alpha=0.7)
            ax3.axvline(x=mid_t, color="k", linestyle=":", alpha=0.5,
                        label="Imagination start")
            ax3.set_xlabel("Timestep")
            ax3.set_ylabel("KL(post || prior)")
            ax3.set_title("Prior-Posterior KL Divergence")
            ax3.legend(fontsize=7)
            ax3.grid(True, alpha=0.3)

            # ---- Panel 4: Trajectory ADE over imagination horizon ----
            ax4 = fig.add_subplot(gs[1, 0])
            if real_trajs is not None:
                nh = 6
                if hn_key is not None:
                    nh = int(batch["observations"][hn_key][i, mid_t].item())
                ade_imag = []
                ade_post = []
                for t in range(min(H, real_trajs.shape[1] - mid_t - 1)):
                    gt_t = real_trajs[i, mid_t + 1 + t]  # (N, T_pred, 2)
                    im_t = imag_trajs_np[i, t]            # (N, T_pred, 2)
                    po_t = post_trajs_np[i, mid_t + 1 + t] if mid_t + 1 + t < T else im_t
                    ade_i_vals, ade_p_vals = [], []
                    for h in range(min(nh, gt_t.shape[0])):
                        if np.all(np.abs(gt_t[h]) > 90):
                            continue
                        ade_i_vals.append(np.linalg.norm(gt_t[h] - im_t[h], axis=-1).mean())
                        ade_p_vals.append(np.linalg.norm(gt_t[h] - po_t[h], axis=-1).mean())
                    ade_imag.append(np.mean(ade_i_vals) if ade_i_vals else 0)
                    ade_post.append(np.mean(ade_p_vals) if ade_p_vals else 0)
                if ade_imag:
                    ax4.plot(range(len(ade_imag)), ade_imag, "r-o", markersize=3,
                             label="Imagined ADE")
                    ax4.plot(range(len(ade_post)), ade_post, "g--s", markersize=3,
                             label="Posterior ADE")
                    ax4.set_xlabel("Steps after mid_t")
                    ax4.set_ylabel("ADE (m)")
                    ax4.set_title("Trajectory ADE: Imagined vs Posterior")
                    ax4.legend(fontsize=7)
                    ax4.grid(True, alpha=0.3)

            # ---- Panel 5: Trajectory at mid_t+1 (GT vs Post vs Imag) ----
            ax5 = fig.add_subplot(gs[1, 1])
            if real_trajs is not None and mid_t + 1 < T:
                nh = 6
                if hn_key is not None:
                    nh = int(batch["observations"][hn_key][i, mid_t].item())
                colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
                for h in range(min(nh, len(colors))):
                    gt_h = real_trajs[i, mid_t + 1, h]
                    if np.all(np.abs(gt_h) > 90):
                        continue
                    c = colors[h]
                    ax5.plot(gt_h[:, 0], gt_h[:, 1], "o-", color=c, alpha=0.3,
                             markersize=5, label=f"H{h} GT" if h == 0 else "")
                    po_h = post_trajs_np[i, mid_t + 1, h]
                    ax5.plot(po_h[:, 0], po_h[:, 1], "s--", color=c, alpha=0.6,
                             markersize=4, label=f"H{h} Post" if h == 0 else "")
                    im_h = imag_trajs_np[i, 0, h]
                    ax5.plot(im_h[:, 0], im_h[:, 1], "^-", color=c, alpha=0.9,
                             markersize=5, label=f"H{h} Imag" if h == 0 else "")
                ax5.set_xlabel("x")
                ax5.set_ylabel("z")
                ax5.set_title(f"Trajectory at t={mid_t+1} (GT/Post/Imag)")
                ax5.legend(fontsize=6)
                ax5.set_aspect("equal")
                ax5.grid(True, alpha=0.3)

            # ---- Panel 6: Trajectory at mid_t + H (long-horizon) ----
            ax6 = fig.add_subplot(gs[1, 2])
            last_t = min(H - 1, real_trajs.shape[1] - mid_t - 2) if real_trajs is not None else 0
            if real_trajs is not None and mid_t + 1 + last_t < T and last_t > 0:
                nh = 6
                if hn_key is not None:
                    nh = int(batch["observations"][hn_key][i, mid_t].item())
                for h in range(min(nh, len(colors))):
                    gt_h = real_trajs[i, mid_t + 1 + last_t, h]
                    if np.all(np.abs(gt_h) > 90):
                        continue
                    c = colors[h]
                    ax6.plot(gt_h[:, 0], gt_h[:, 1], "o-", color=c, alpha=0.3,
                             markersize=5, label=f"H{h} GT" if h == 0 else "")
                    if mid_t + 1 + last_t < T:
                        po_h = post_trajs_np[i, mid_t + 1 + last_t, h]
                        ax6.plot(po_h[:, 0], po_h[:, 1], "s--", color=c, alpha=0.6,
                                 markersize=4, label=f"H{h} Post" if h == 0 else "")
                    im_h = imag_trajs_np[i, last_t, h]
                    ax6.plot(im_h[:, 0], im_h[:, 1], "^-", color=c, alpha=0.9,
                             markersize=5, label=f"H{h} Imag" if h == 0 else "")
                ax6.set_xlabel("x")
                ax6.set_ylabel("z")
                ax6.set_title(f"Trajectory at t={mid_t+1+last_t} (GT/Post/Imag)")
                ax6.legend(fontsize=6)
                ax6.set_aspect("equal")
                ax6.grid(True, alpha=0.3)

            # ---- Row 3: Depth comparison (GT / Posterior / Imagined) ----
            if depth_key is not None:
                import cv2 as _cv2
                gt_depth_all = batch["observations"][depth_key][i].cpu().numpy()

                def _depth_to_img(d):
                    """Convert depth array to displayable 2D image."""
                    if d.ndim == 3:
                        d = d[..., 0] if d.shape[-1] == 1 else d[0]
                    if d.max() > 1.0:
                        d = d.astype(np.float64) / 255.0
                    d = np.clip(d, 0, 1)
                    return d

                depth_steps = [mid_t, min(mid_t + 1, T - 1), min(mid_t + H // 2, T - 1)]
                titles_row3 = [
                    f"GT Depth t={depth_steps[0]}",
                    f"Posterior t={depth_steps[0]} vs Imag t={depth_steps[1]}",
                    f"Imag Depth t={depth_steps[2]}",
                ]

                # Panel 7: GT depth at mid_t + Posterior depth at mid_t
                ax7 = fig.add_subplot(gs[2, 0])
                gt_d = _depth_to_img(gt_depth_all[mid_t])
                po_d = _depth_to_img(post_depths_np[i, mid_t])
                if gt_d.shape != po_d.shape:
                    po_d = _cv2.resize(po_d, (gt_d.shape[1], gt_d.shape[0]))
                combined = np.concatenate([gt_d, po_d], axis=1)
                ax7.imshow(combined, cmap="turbo", vmin=0, vmax=1)
                ax7.set_title(f"GT (left) vs Posterior (right) at t={mid_t}")
                ax7.axis("off")

                # Panel 8: GT depth at mid_t+1 vs Imagined depth step 0
                # Use prior-only (no skip) imag depth so different steps look different
                use_imag = imag_depths_no_skip_np if imag_depths_no_skip_np is not None else imag_depths_np
                ax8 = fig.add_subplot(gs[2, 1])
                gt_d1 = _depth_to_img(gt_depth_all[min(mid_t + 1, T - 1)])
                im_d0 = _depth_to_img(use_imag[i, 0])
                if gt_d1.shape != im_d0.shape:
                    im_d0 = _cv2.resize(im_d0, (gt_d1.shape[1], gt_d1.shape[0]))
                combined1 = np.concatenate([gt_d1, im_d0], axis=1)
                ax8.imshow(combined1, cmap="turbo", vmin=0, vmax=1)
                ax8.set_title(
                    f"GT t={mid_t+1} (L) vs Imag step 0 (R)"
                    + (" [prior only]" if imag_depths_no_skip_np is not None else "")
                )
                ax8.axis("off")

                # Panel 9: GT vs Imagined depth at multiple horizon steps (prior-only for variation)
                ax9 = fig.add_subplot(gs[2, 2])
                show_steps = [0, H // 4, H // 2, H - 1]
                show_steps = [s for s in show_steps if s < H]
                depth_strips = []
                for s in show_steps:
                    gt_t = min(mid_t + 1 + s, T - 1)
                    gt_ds = _depth_to_img(gt_depth_all[gt_t])
                    im_ds = _depth_to_img(use_imag[i, s])
                    if gt_ds.shape != im_ds.shape:
                        im_ds = _cv2.resize(im_ds, (gt_ds.shape[1], gt_ds.shape[0]))
                    pair = np.concatenate([gt_ds, im_ds], axis=1)
                    h_strip, w_strip = pair.shape[:2]
                    target_h = 64
                    ratio = target_h / h_strip
                    pair = _cv2.resize(pair, (int(w_strip * ratio), target_h))
                    depth_strips.append(pair)
                if depth_strips:
                    strip = np.concatenate(depth_strips, axis=0)
                    ax9.imshow(strip, cmap="turbo", vmin=0, vmax=1)
                    step_labels = ", ".join([str(s) for s in show_steps])
                    ax9.set_title(
                        f"GT|Imag Depth steps {step_labels}"
                        + (" [prior only]" if imag_depths_no_skip_np is not None else "")
                    )
                ax9.axis("off")

            fig.suptitle(
                f"Imagination vs Reality — Sample {i} "
                f"(observe t=0..{mid_t}, imagine t={mid_t+1}..{mid_t+H})",
                fontsize=13,
            )
            save_path = os.path.join(vis_dir, f"step{step:08d}_sample{i}.png")
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            if writer is not None:
                import cv2
                img = cv2.imread(save_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
                    writer.add_image(
                        f"imag_vs_real/sample_{i}", img_tensor, step
                    )

        # --- Summary statistics ---
        real_future = real_rewards[:num_samples, mid_t + 1:mid_t + 1 + H, 0]
        imag_future = imag_rewards_np[:num_samples, :H, 0]
        min_len = min(real_future.shape[1], imag_future.shape[1])
        if min_len > 0:
            reward_mae = np.abs(
                imag_future[:, :min_len] - real_future[:, :min_len]
            ).mean()
            reward_corr_vals = []
            for si in range(num_samples):
                if np.std(real_future[si, :min_len]) > 1e-6:
                    corr = np.corrcoef(
                        real_future[si, :min_len], imag_future[si, :min_len]
                    )[0, 1]
                    if not np.isnan(corr):
                        reward_corr_vals.append(corr)
            reward_corr = np.mean(reward_corr_vals) if reward_corr_vals else float("nan")
        else:
            reward_mae = float("nan")
            reward_corr = float("nan")

        # Depth RMSE: imagined vs real
        depth_rmse = float("nan")
        if depth_key is not None:
            gt_depth_all_np = batch["observations"][depth_key].cpu().numpy()
            rmse_vals = []
            for si in range(num_samples):
                for t in range(min(H, gt_depth_all_np.shape[1] - mid_t - 1)):
                    gt_d = gt_depth_all_np[si, mid_t + 1 + t]
                    im_d = imag_depths_np[si, t]
                    if gt_d.ndim == 3:
                        gt_d = gt_d[..., 0] if gt_d.shape[-1] == 1 else gt_d[0]
                    if im_d.ndim == 3:
                        im_d = im_d[..., 0] if im_d.shape[-1] == 1 else im_d[0]
                    if gt_d.shape != im_d.shape:
                        import cv2 as _cv2_rmse
                        im_d = _cv2_rmse.resize(im_d, (gt_d.shape[1], gt_d.shape[0]))
                    rmse_vals.append(float(np.sqrt(np.mean((gt_d - im_d) ** 2))))
            if rmse_vals:
                depth_rmse = float(np.mean(rmse_vals))

        logger.info(
            f"[Imag vs Real] step={step}, H={H}, mid_t={mid_t}, "
            f"reward_MAE={reward_mae:.4f}, reward_corr={reward_corr:.3f}, "
            f"mean_KL={kl_np[:num_samples].mean():.4f}, depth_RMSE={depth_rmse:.4f}"
        )

        if writer is not None:
            writer.add_scalar("imag_vs_real/reward_MAE", reward_mae, step)
            if not np.isnan(reward_corr):
                writer.add_scalar("imag_vs_real/reward_corr", reward_corr, step)
            writer.add_scalar("imag_vs_real/mean_KL", kl_np[:num_samples].mean(), step)
            if not np.isnan(depth_rmse):
                writer.add_scalar("imag_vs_real/depth_RMSE", depth_rmse, step)

        logger.info(f"[Imag vs Real] Saved {num_samples} figure(s) to {vis_dir}")

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )

        with g_timer.avg_time("trainer.sample_action"), inference_mode():
            # Sample actions
            step_batch = self._agent.rollouts.get_current_step(
                env_slice, buffer_index
            )

            profiling_wrapper.range_push("compute actions")

            # Obtain lenghts
            step_batch_lens = {
                k: v
                for k, v in step_batch.items()
                if k.startswith("index_len")
            }
            action_data = self._agent.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
                **step_batch_lens,
            )

        profiling_wrapper.range_pop()  # compute actions

        with g_timer.avg_time("trainer.obs_insert"):
            _exhausted = getattr(self, "_envs_exhausted", set())
            for index_env, act in zip(
                range(env_slice.start, env_slice.stop),
                action_data.env_actions.cpu().unbind(0),
            ):
                if index_env in _exhausted:
                    continue
                if hasattr(self._agent, '_agents') and self._agent._agents[0]._actor_critic.action_distribution_type == 'categorical':
                    act = act.numpy()
                elif is_continuous_action_space(self._env_spec.action_space):
                    # Clipping actions to the specified limits
                    act = np.clip(
                        act.numpy(),
                        self._env_spec.action_space.low,
                        self._env_spec.action_space.high,
                    )
                else:
                    act = act.item()
                self.envs.async_step_at(index_env, act)

        with g_timer.avg_time("trainer.obs_insert"):
            self._agent.rollouts.insert(
                next_recurrent_hidden_states=action_data.rnn_hidden_states,
                actions=action_data.actions,
                action_log_probs=action_data.action_log_probs,
                value_preds=action_data.values,
                wm_features=action_data.wm_features,
                lookahead_features=action_data.lookahead_features,
                buffer_index=buffer_index,
                should_inserts=action_data.should_inserts,
                action_data=action_data,
            )

    def _collect_environment_result(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )

        with g_timer.avg_time("trainer.step_env"):
            _exhausted = getattr(self, "_envs_exhausted", set())
            _is_collect = getattr(self, "collect_replay_only", False)
            if not hasattr(self, "_last_obs_per_env"):
                self._last_obs_per_env = {}

            outputs = []
            for index_env in range(env_slice.start, env_slice.stop):
                if index_env in _exhausted:
                    outputs.append(None)
                else:
                    outputs.append(self.envs.wait_step_at(index_env))

            _newly_exhausted = set()
            _any_valid_obs = None
            for local_i, index_env in enumerate(
                range(env_slice.start, env_slice.stop)
            ):
                if outputs[local_i] is not None:
                    obs_i, rew_i, done_i, info_i = outputs[local_i]
                    if _any_valid_obs is None:
                        _any_valid_obs = obs_i
                    self._last_obs_per_env[index_env] = obs_i
                    if info_i.get("_episodes_exhausted", False):
                        _newly_exhausted.add(index_env)

            if _newly_exhausted and _is_collect:
                _exhausted.update(_newly_exhausted)
                logger.info(
                    f"Environments exhausted: {sorted(_newly_exhausted)} "
                    f"(total {len(_exhausted)}/{self.envs.num_envs})"
                )
                if len(_exhausted) >= self.envs.num_envs:
                    self._all_envs_exhausted = True

            _any_valid_info = None
            for o in outputs:
                if o is not None and o[3]:
                    _any_valid_info = o[3]
                    break

            filled_outputs = []
            for local_i, index_env in enumerate(
                range(env_slice.start, env_slice.stop)
            ):
                if outputs[local_i] is not None:
                    filled_outputs.append(outputs[local_i])
                else:
                    dummy_obs = self._last_obs_per_env.get(index_env, _any_valid_obs)
                    if dummy_obs is None:
                        for eidx in self._last_obs_per_env:
                            dummy_obs = self._last_obs_per_env[eidx]
                            break
                    dummy_info = {}
                    if _any_valid_info is not None:
                        for k, v in _any_valid_info.items():
                            if isinstance(v, (int, float, bool)):
                                dummy_info[k] = type(v)(0)
                            elif isinstance(v, str):
                                dummy_info[k] = ""
                            elif isinstance(v, np.ndarray) and np.size(v) == 1:
                                dummy_info[k] = type(v)(0)
                    filled_outputs.append((dummy_obs, 0.0, False, dummy_info))

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*filled_outputs)
            ]

        with g_timer.avg_time("trainer.update_stats"):
            observations = self.envs.post_step(observations)

            # --- Collect mode: cache raw visual obs & infos, then strip
            #     RGB keys so they don't enter rollout storage / visual encoder.
            _is_collect = getattr(self, "collect_replay_only", False)
            _strip_keys = getattr(self, "_collect_visual_strip_keys", [])
            if _is_collect:
                if not hasattr(self, "_collect_step_cache"):
                    self._collect_step_cache = {}
                if not hasattr(self, "_episode_done_metrics"):
                    self._episode_done_metrics = {}

                _exhaust_set = getattr(self, "_envs_exhausted", set())
                for local_i, global_i in enumerate(
                    range(env_slice.start, env_slice.stop)
                ):
                    if global_i in _exhaust_set:
                        continue
                    if global_i not in self._collect_step_cache:
                        self._collect_step_cache[global_i] = []
                    raw_obs_i = observations[local_i]
                    cache_obs = {}
                    _wm_cfg = self.config.habitat_baselines.get("world_model", None)
                    _save_all_robots = (
                        _wm_cfg is not None
                        and getattr(_wm_cfg, "save_all_robots", False)
                    )
                    _collect_num_robots = (
                        getattr(_wm_cfg, "num_robots", 1)
                        if _wm_cfg is not None else 1
                    )
                    _robot_prefixes = tuple(
                        f"agent_{i}_" for i in range(_collect_num_robots)
                    ) if _save_all_robots else ("agent_0_",)
                    for k, v in raw_obs_i.items():
                        if k.startswith("agent_") and not k.startswith(_robot_prefixes):
                            continue
                        if isinstance(v, np.ndarray) and v.ndim >= 2:
                            cache_obs[k] = v.copy()
                        elif isinstance(v, torch.Tensor) and v.ndim >= 2:
                            cache_obs[k] = v.detach().cpu().numpy().copy()
                    cache_info = {}
                    info_i = infos[local_i]
                    if "top_down_map" in info_i:
                        td = info_i["top_down_map"]
                        cache_info["top_down_map"] = {
                            kk: (vv.copy() if isinstance(vv, np.ndarray) else vv)
                            for kk, vv in td.items()
                        }
                    if "collisions" in info_i:
                        cache_info["collisions"] = dict(info_i["collisions"])
                    self._collect_step_cache[global_i].append({
                        "observations": cache_obs,
                        "info": cache_info,
                    })

                # Strip RGB keys from observations before batch_obs so they
                # don't enter rollout storage or the visual encoder.
                if _strip_keys:
                    for obs_dict in observations:
                        for vk in _strip_keys:
                            obs_dict.pop(vk, None)

            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            # Check for per-robot rewards from multi-robot training
            _shared_policy = getattr(
                self.config.habitat_baselines.rl.agent, "shared_policy", False
            )
            _num_robots = getattr(
                self.config.habitat_baselines.rl.agent, "num_robots", 1
            )
            if _shared_policy and _num_robots > 1:
                # Build [batch, num_robots] reward tensor
                per_robot_rewards = []
                for local_i, info_i in enumerate(infos):
                    prr = info_i.get("per_robot_rewards", None)
                    if prr is not None:
                        # Pad or truncate to _num_robots
                        prr = list(prr[:_num_robots])
                        while len(prr) < _num_robots:
                            prr.append(0.0)
                        per_robot_rewards.append(prr)
                    else:
                        per_robot_rewards.append([rewards_l[local_i]] * _num_robots)
                rewards = torch.tensor(
                    per_robot_rewards,
                    dtype=torch.float,
                    device=self.current_episode_reward.device,
                )
            else:
                rewards = torch.tensor(
                    rewards_l,
                    dtype=torch.float,
                    device=self.current_episode_reward.device,
                )
                rewards = rewards.unsqueeze(1)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device=self.current_episode_reward.device,
            )
            done_masks = torch.logical_not(not_done_masks)

            # For episode stats tracking, use agent_0's reward (first column)
            rewards_for_stats = rewards[:, :1] if rewards.dim() == 2 and rewards.shape[-1] > 1 else rewards
            self.current_episode_reward[env_slice] += rewards_for_stats
            current_ep_reward = self.current_episode_reward[env_slice]
            self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
            self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore

            _ref_info = next((inf for inf in infos if inf), infos[0])
            self._single_proc_infos = extract_scalars_from_infos(
                infos,
                ignore_keys=set(
                    k for k in _ref_info.keys() if k not in self._rank0_keys
                ),
            )
            _ignore = set(self._rank0_keys)
            _ignore.add("per_robot_rewards")
            extracted_infos = extract_scalars_from_infos(
                infos, ignore_keys=_ignore
            )
            _expected_len = env_slice.stop - env_slice.start
            for k, v_k in extracted_infos.items():
                if len(v_k) != _expected_len:
                    continue
                v = torch.tensor(
                    v_k,
                    dtype=torch.float,
                    device=self.current_episode_reward.device,
                ).unsqueeze(1)
                if k not in self.running_episode_stats:
                    self.running_episode_stats[k] = torch.zeros_like(
                        self.running_episode_stats["count"]
                    )
                self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

            self.current_episode_reward[env_slice].masked_fill_(
                done_masks, 0.0
            )

            if _is_collect:
                _metric_keys = ("spl", "success", "distance_to_goal", "human_collision")
                if any(dones):
                    try:
                        episodes = self.envs.current_episodes()
                    except Exception:
                        episodes = []
                else:
                    episodes = []
                _n_done_this_step = 0
                for local_i, global_i in enumerate(
                    range(env_slice.start, env_slice.stop)
                ):
                    if dones[local_i]:
                        _n_done_this_step += 1
                        m = {}
                        info_i = infos[local_i] if local_i < len(infos) else {}
                        for mk in _metric_keys:
                            if mk in info_i:
                                v = info_i[mk]
                                if isinstance(v, (int, float, bool)):
                                    m[mk] = float(v)
                        m["reward"] = current_ep_reward[local_i].item()
                        if global_i < len(episodes):
                            ep = episodes[global_i]
                            m["episode_id"] = getattr(ep, "episode_id", None)
                            m["scene_id"] = getattr(ep, "scene_id", None)
                        self._episode_done_metrics[global_i] = m
        # Key Modification between the trainer and the original ppo trainer
        if self._is_static_encoder:
            self._encoder = self._agent.actor_critic.visual_encoder
            if self._encoder is None:
                self._encoder = self._agent._agents[0].actor_critic.visual_encoder
                with inference_mode(), g_timer.avg_time("trainer.visual_features"):
                    batch_temp = {key.replace('agent_0_', ''): value for key, value in batch.items()}
                    batch[
                        'agent_0_' + PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch_temp)
            else:
                with inference_mode(), g_timer.avg_time("trainer.visual_features"):
                    batch[
                        PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch)
        self._agent.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self._agent.rollouts.advance_rollout(buffer_index)

        return env_slice.stop - env_slice.start

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(self):
        self._compute_actions_and_step_envs()
        return self._collect_environment_result()

    def _inject_bootstrap_wm_cached_feature(self, step_batch):
        """Inject rollout-cached WM / lookahead features for last-step value bootstrap."""
        wm_key = "wm_cached_feature"
        la_key = "lookahead_cached_feature"
        rollouts = self._agent.rollouts

        # Single-agent storage path.
        if hasattr(rollouts, "buffers"):
            step_idx = rollouts.current_rollout_step_idx
            if step_idx > 0:
                if "wm_features" in rollouts.buffers:
                    step_batch["observations"][wm_key] = rollouts.buffers[
                        "wm_features"
                    ][step_idx - 1]
                if "lookahead_features" in rollouts.buffers:
                    step_batch["observations"][la_key] = rollouts.buffers[
                        "lookahead_features"
                    ][step_idx - 1]
            return

        # Multi-agent storage path.
        if hasattr(rollouts, "_active_storages"):
            for agent_i, storage in enumerate(rollouts._active_storages):
                if storage is None or not hasattr(storage, "buffers"):
                    continue
                step_idx = storage.current_rollout_step_idx
                if step_idx > 0:
                    if "wm_features" in storage.buffers:
                        step_batch["observations"][
                            f"agent_{agent_i}_{wm_key}"
                        ] = storage.buffers["wm_features"][step_idx - 1]
                    if "lookahead_features" in storage.buffers:
                        step_batch["observations"][
                            f"agent_{agent_i}_{la_key}"
                        ] = storage.buffers["lookahead_features"][step_idx - 1]

    @profiling_wrapper.RangeContext("_update_agent")
    @g_timer.avg_time("trainer.update_agent")
    def _update_agent(self):
        with inference_mode():
            step_batch = self._agent.rollouts.get_last_step()
            self._inject_bootstrap_wm_cached_feature(step_batch)
            step_batch_lens = {
                k: v
                for k, v in step_batch.items()
                if k.startswith("index_len")
            }

            next_value = self._agent.actor_critic.get_value(
                step_batch["observations"],
                step_batch.get("recurrent_hidden_states", None),
                step_batch["prev_actions"],
                step_batch["masks"],
                **step_batch_lens,
            )

        self._agent.rollouts.compute_returns(
            next_value,
            self._ppo_cfg.use_gae,
            self._ppo_cfg.gamma,
            self._ppo_cfg.tau,
        )

        self._agent.train()

        # Decoupled mode: ensure PPO update never touches WM params.
        if self.use_world_model:
            self._set_world_model_grad_enabled(False)
            self._clear_world_model_grads()

        losses = self._agent.updater.update(self._agent.rollouts)

        self._agent.rollouts.after_update()
        self._agent.after_update()

        if self.use_world_model:
            self._clear_world_model_grads()

        return losses

    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats_before_reduce = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )

        stats = self._all_reduce(stats_before_reduce)

        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i].clone())

        # ===== DEBUG: Coalesce diagnostics (commented out) =====
        # _resume_n = getattr(self, '_resume_update_num', 0)
        # _updates_since_resume = self.num_updates_done - _resume_n
        # _debug_should_log = (
        #     _updates_since_resume <= 5
        #     or self.num_updates_done % 10 == 0
        # )
        # if _debug_should_log:
        #     _debug_keys = ["count", "spl", "success", "reward", "distance_to_goal"]
        #     if self._is_distributed:
        #         _world_size = torch.distributed.get_world_size()
        #     else:
        #         _world_size = 1
        #     logger.info(
        #         f"[COALESCE DEBUG] update={self.num_updates_done} "
        #         f"(+{_updates_since_resume} since resume) "
        #         f"world_size={_world_size}, count_steps_delta={count_steps_delta}"
        #     )
        #     for k in _debug_keys:
        #         _idx = stats_ordering.index(k) if k in stats_ordering else -1
        #         if _idx < 0:
        #             continue
        #         _before = stats_before_reduce[_idx].sum().item()
        #         _after = stats[_idx].sum().item()
        #         v = self.window_episode_stats[k]
        #         _prev_sum = v[-2].sum().item() if len(v) > 1 else 0
        #         _this_increment = _after - _prev_sum
        #         logger.info(
        #             f"  {k}: local_sum={_before:.4f}, "
        #             f"all_reduce_sum={_after:.4f}, "
        #             f"prev_snapshot={_prev_sum:.4f}, "
        #             f"increment={_this_increment:.4f}, "
        #             f"window_len={len(v)}"
        #         )

        if self._is_distributed:
            loss_name_ordering = sorted(losses.keys())
            stats = torch.tensor(
                [losses[k] for k in loss_name_ordering] + [count_steps_delta],
                device="cpu",
                dtype=torch.float32,
            )
            stats = self._all_reduce(stats)
            count_steps_delta = int(stats[-1].item())
            stats /= torch.distributed.get_world_size()

            losses = {
                k: stats[i].item() for i, k in enumerate(loss_name_ordering)
            }

        if self._is_distributed and rank0_only():
            self.num_rollouts_done_store.set("num_done", "0")

        self.num_steps_done += count_steps_delta

        return losses

    @rank0_only
    def _training_log(
        self,
        writer,
        losses: Dict[str, float],
        prev_time: int = 0,
        count_checkpoints: Optional[int] = None,
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }

        # ===== FIX: Resume 后平滑过渡 =====
        # resume 后窗口从小开始增长，早期完成的 episode 以失败居多，
        # 导致指标比真实策略水平低。用旧窗口平均值做加权混合来平滑。
        # 权重 alpha = 当前窗口大小 / 目标窗口大小，alpha 从 0 → 1。
        # smoothed = alpha * raw + (1 - alpha) * old_avg
        _old_avg = getattr(self, '_resume_old_window_avg', {})
        _target_wsize = self._ppo_cfg.reward_window_size
        _cur_wsize = len(self.window_episode_stats.get("count", []))
        if _old_avg and _cur_wsize < _target_wsize:
            _alpha = _cur_wsize / _target_wsize
            for k in metrics:
                if k in _old_avg:
                    metrics[k] = _alpha * metrics[k] + (1 - _alpha) * _old_avg[k]
            if "reward" in _old_avg:
                _raw_reward = deltas["reward"] / deltas["count"]
                _smoothed_reward = _alpha * _raw_reward + (1 - _alpha) * _old_avg["reward"]
            else:
                _smoothed_reward = deltas["reward"] / deltas["count"]
        else:
            _smoothed_reward = deltas["reward"] / deltas["count"]

        writer.add_scalar(
            "reward",
            _smoothed_reward,
            self.num_steps_done,
        )

        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
        for k, v in losses.items():
            writer.add_scalar(f"learner/{k}", v, self.num_steps_done)

        for k, v in self._single_proc_infos.items():
            writer.add_scalar(k, np.mean(v), self.num_steps_done)

        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)

        # Log perf metrics.
        writer.add_scalar("perf/fps", fps, self.num_steps_done)

        for timer_name, timer_val in g_timer.items():
            writer.add_scalar(
                f"perf/{timer_name}",
                timer_val.mean,
                self.num_steps_done,
            )

        # log stats
        if (
            self.num_updates_done % self.config.habitat_baselines.log_interval
            == 0
        ):
            logger.info("")

            # Calculate progress and timing
            progress_pct = self.percent_done() * 100
            total_steps = self.config.habitat_baselines.total_num_steps
            remaining_steps = max(total_steps - self.num_steps_done, 0)
            elapsed_time = (time.time() - self.t_start) + prev_time

            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    fps,
                )
            )

            logger.info(
                f"Num updates: {self.num_updates_done}\tNum frames: {self.num_steps_done}"
            )

            logger.info(
                f"Progress: {progress_pct:.2f}% ({self.num_steps_done}/{total_steps} steps) | "
                f"Remaining: {remaining_steps} steps"
            )

            if self.num_steps_done > 0:
                time_per_step = elapsed_time / self.num_steps_done
                estimated_remaining_time = time_per_step * remaining_steps
                hours_remaining = estimated_remaining_time / 3600
                logger.info(
                    f"Elapsed: {elapsed_time/3600:.2f}h | Estimated remaining: "
                    f"{hours_remaining:.2f}h ({hours_remaining/24:.2f} days)"
                )

            # Calculate next checkpoint progress
            if self.config.habitat_baselines.num_checkpoints != -1:
                num_ckpts = self.config.habitat_baselines.num_checkpoints
                checkpoint_interval_steps = total_steps / num_ckpts
                next_segment_1based = int(
                    self.num_steps_done / checkpoint_interval_steps
                ) + 1
                next_checkpoint_steps = (
                    next_segment_1based * checkpoint_interval_steps
                )
                steps_to_next_ckpt = max(
                    int(next_checkpoint_steps - self.num_steps_done), 0
                )
                pct_to_next_ckpt = (
                    (self.num_steps_done % checkpoint_interval_steps)
                    / checkpoint_interval_steps
                    * 100
                )
                # 显示实际将要保存的文件名 (count_checkpoints)，与 "Saved ckpt.X.pth" 一致
                next_save_str = (
                    f"Next save: ckpt.{count_checkpoints}.pth"
                    if count_checkpoints is not None
                    else f"Next segment: {next_segment_1based}/{num_ckpts}"
                )
                logger.info(
                    f"{next_save_str} | "
                    f"Progress to next: {pct_to_next_ckpt:.1f}% "
                    f"({steps_to_next_ckpt} steps remaining)"
                )

            if self.train_world_model:
                logger.info(
                    "WM replay status: "
                    f"{self._get_replay_buffer_size()}/{self.replay_buffer_size} "
                    f"(warmup={self.replay_buffer_warmup})"
                )
            elif getattr(self, "collect_replay_only", False):
                _n_written = getattr(self, "_collect_episodes_written", 0)
                _n_discarded = getattr(self, "_collect_episodes_discarded", 0)
                _ep_target = getattr(self, "collect_replay_episodes", 0)
                _target_str = str(_ep_target) if _ep_target > 0 else "unlimited"
                logger.info(
                    f"Collect status: {_n_written} saved, "
                    f"{_n_discarded} discarded (non-success), "
                    f"target={_target_str} total (this rank)"
                )

            logger.info("  --- Losses ---")
            for k, v in sorted(losses.items()):
                logger.info(f"    {k}: {v:.4f}")

            _is_smoothing = bool(_old_avg and _cur_wsize < _target_wsize)
            if _is_smoothing:
                logger.info(
                    "  --- 窗口指标 (window_size={}, smoothing alpha={:.2f}) ---".format(
                        _cur_wsize, _alpha
                    )
                )
            else:
                logger.info(
                    "  --- 窗口指标 (window_size={}) ---".format(
                        len(self.window_episode_stats["count"])
                    )
                )
            logger.info(f"    count: {deltas['count']:.4f}")
            for k in sorted(metrics.keys()):
                _raw = deltas[k] / deltas["count"] if k in deltas else 0
                if _is_smoothing and k in _old_avg:
                    logger.info(f"    {k}: {metrics[k]:.4f} (raw={_raw:.4f}, old={_old_avg[k]:.4f})")
                else:
                    logger.info(f"    {k}: {metrics[k]:.4f}")

            # ===== DEBUG: Window detail for key metrics (commented out) =====
            # _debug_detail_keys = ["count", "spl", "success", "distance_to_goal", "reward"]
            # _resume_n = getattr(self, '_resume_update_num', 0)
            # _updates_since_resume = self.num_updates_done - _resume_n
            # logger.info(f"  --- [DEBUG] 窗口详情 (update={self.num_updates_done}, "
            #             f"+{_updates_since_resume} since resume) ---")
            # for k in _debug_detail_keys:
            #     if k in self.window_episode_stats:
            #         v = self.window_episode_stats[k]
            #         wlen = len(v)
            #         first_sum = v[0].sum().item() if wlen > 0 else 0
            #         last_sum = v[-1].sum().item() if wlen > 0 else 0
            #         delta = (v[-1] - v[0]).sum().item() if wlen > 1 else (v[0].sum().item() if wlen > 0 else 0)
            #         last_incr = (v[-1] - v[-2]).sum().item() if wlen > 1 else 0
            #         logger.info(
            #             f"    {k}: wlen={wlen}, "
            #             f"first={first_sum:.4f}, last={last_sum:.4f}, "
            #             f"delta={delta:.4f}, "
            #             f"last_incr={last_incr:.4f}, "
            #             f"avg={delta / max(deltas['count'], 1.0):.4f}"
            #         )
            #         if wlen <= 6:
            #             all_sums = [vi.sum().item() for vi in v]
            #             logger.info(f"      all_snapshots: {[f'{s:.2f}' for s in all_sums]}")
            #         elif wlen <= 10:
            #             all_sums = [vi.sum().item() for vi in v]
            #             logger.info(f"      snapshots: {[f'{s:.1f}' for s in all_sums]}")
            #         else:
            #             head_sums = [v[j].sum().item() for j in range(3)]
            #             tail_sums = [v[j].sum().item() for j in range(wlen-3, wlen)]
            #             logger.info(
            #                 f"      first_3: {[f'{s:.2f}' for s in head_sums]}, "
            #                 f"last_3: {[f'{s:.2f}' for s in tail_sums]}"
            #             )

            logger.info("  --- Perf (mean) ---")
            for k, v in g_timer.items():
                logger.info(f"    {k}: {v.mean:.3f}s")

            if self.config.habitat_baselines.should_log_single_proc_infos:
                logger.info("  --- Single-proc infos ---")
                for k, v in self._single_proc_infos.items():
                    logger.info(f"    {k}: {np.mean(v):.4f}")

            if torch.cuda.is_available():
                try:
                    alloc = torch.cuda.memory_allocated() / (1024**3)
                    reserved = torch.cuda.memory_reserved() / (1024**3)
                    logger.info(
                        "  --- GPU ---  "
                        f"alloc: {alloc:.2f} GB  |  reserved: {reserved:.2f} GB"
                    )
                except Exception:
                    pass

            logger.info("=" * 70)

    def should_end_early(self, rollout_step) -> bool:
        if not self._is_distributed:
            return False
        # This is where the preemption of workers happens.  If a
        # worker detects it will be a straggler, it preempts itself!
        return (
            rollout_step
            >= self.config.habitat_baselines.rl.ppo.num_steps
            * self.SHORT_ROLLOUT_THRESHOLD
        ) and int(self.num_rollouts_done_store.get("num_done")) >= (
            self.config.habitat_baselines.rl.ddppo.sync_frac
            * torch.distributed.get_world_size()
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        resume_state = load_resume_state(self.config)
        self._init_train(resume_state)

        count_checkpoints = 0
        prev_time = 0

        if self._is_distributed:
            torch.distributed.barrier()

        resume_run_id = None
        if resume_state is not None:
            self._agent.load_state_dict(resume_state)

            requeue_stats = resume_state["requeue_stats"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            # ===== FIX: Resume 后窗口指标连续性修复 =====
            # 不恢复 running_episode_stats 和 window_episode_stats。
            # 原因：resume 后环境被完全重建，所有 episode 从头开始，
            # running_episode_stats 的旧累计值与新环境状态不匹配。
            # 让两者都保持 _init_train 中初始化的零值，与从头训练一致。
            # 这样窗口指标忠实反映 resume 后的真实策略表现。
            #
            # 注意：不恢复 running_episode_stats 意味着旧累计数据丢失，
            # 但窗口指标只关心增量（v[-1] - v[0]），不依赖历史绝对值。
            # logger.info("[RESUME] NOT restoring running_episode_stats and "
            #             "window_episode_stats (reset to zeros, consistent with "
            #             "newly rebuilt environments)")

            # 保存旧窗口平均值用于平滑过渡
            _old_window = requeue_stats.get("window_episode_stats", {})
            _old_window_len = len(next(iter(_old_window.values()), []))
            self._resume_old_window_avg = {}
            if _old_window_len > 1 and "count" in _old_window:
                _old_count_delta = (_old_window['count'][-1] - _old_window['count'][0]).sum().item()
                if _old_count_delta > 0:
                    for _ok, _ov in _old_window.items():
                        if _ok == "count":
                            continue
                        _ov_delta = (_ov[-1] - _ov[0]).sum().item()
                        self._resume_old_window_avg[_ok] = _ov_delta / _old_count_delta
                # logger.info(f"[RESUME] Old window averages (for reference, {_old_window_len} snapshots):")
                # for _ok in ["count", "spl", "success", "reward", "distance_to_goal"]:
                #     if _ok in self._resume_old_window_avg:
                #         logger.info(f"  {_ok}: {self._resume_old_window_avg[_ok]:.4f}")
                #     elif _ok == "count":
                #         logger.info(f"  count: delta={_old_count_delta:.0f}")

            resume_run_id = requeue_stats.get("run_id", None)

            self._resume_update_num = self.num_updates_done

            # ===== DEBUG: Resume diagnostics (commented out) =====
            # logger.info("=" * 70)
            # logger.info(f"[RESUME DEBUG] num_updates_done={self.num_updates_done}, "
            #             f"num_steps_done={self.num_steps_done}, "
            #             f"num_envs={self.envs.num_envs}, "
            #             f"is_distributed={self._is_distributed}")
            # logger.info("[RESUME DEBUG] running_episode_stats: reset to zeros "
            #             f"(num_envs={self.envs.num_envs})")
            # logger.info("[RESUME DEBUG] window_episode_stats: empty")
            # logger.info("=" * 70)

        with (
            get_writer(
                self.config,
                resume_run_id=resume_run_id,
                flush_secs=self.flush_secs,
                purge_step=int(self.num_steps_done),
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                self._agent.pre_rollout()

                # Resume state on preemption signal only (resume 与 ckpt 同频，见下方 save_checkpoint 处)
                if rank0_only() and SAVE_STATE.is_set():
                    requeue_stats = dict(
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                        run_id=writer.get_run_id(),
                    )
                    save_resume_state(
                        dict(
                            **self._agent.get_resume_state(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                self._agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                with g_timer.avg_time("trainer.rollout_collect"):
                    for buffer_index in range(self._agent.nbuffers):
                        self._compute_actions_and_step_envs(buffer_index)

                    for step in range(self._ppo_cfg.num_steps):
                        is_last_step = (
                            self.should_end_early(step + 1)
                            or (step + 1) == self._ppo_cfg.num_steps
                        )

                        for buffer_index in range(self._agent.nbuffers):
                            count_steps_delta += (
                                self._collect_environment_result(buffer_index)
                            )

                            if (buffer_index + 1) == self._agent.nbuffers:
                                profiling_wrapper.range_pop()  # _collect_rollout_step

                            if not is_last_step:
                                if (buffer_index + 1) == self._agent.nbuffers:
                                    profiling_wrapper.range_push(
                                        "_collect_rollout_step"
                                    )

                                self._compute_actions_and_step_envs(
                                    buffer_index
                                )

                        if getattr(self, "_all_envs_exhausted", False):
                            break
                        if is_last_step:
                            break

                profiling_wrapper.range_pop()  # rollouts loop

                # ==================== Store rollouts to replay buffer ====================
                _should_store = self.train_world_model or getattr(self, "collect_replay_only", False)
                if _should_store:
                    self._store_rollout_to_buffer(self._agent.rollouts)

                # Check if all environments have exhausted their episodes (across all ranks)
                if getattr(self, "collect_replay_only", False):
                    _local_exhausted = 1 if getattr(self, "_all_envs_exhausted", False) else 0
                    if torch.distributed.is_initialized():
                        _exhaust_t = torch.tensor([_local_exhausted], dtype=torch.long, device=self.device)
                        torch.distributed.all_reduce(_exhaust_t, op=torch.distributed.ReduceOp.MIN)
                        _global_all_exhausted = int(_exhaust_t.item()) > 0
                    else:
                        _global_all_exhausted = _local_exhausted > 0

                    if _global_all_exhausted:
                        _written = getattr(self, "_collect_episodes_written", 0)
                        _disc = getattr(self, "_collect_episodes_discarded", 0)
                        if torch.distributed.is_initialized():
                            _written_t = torch.tensor([_written], dtype=torch.long, device=self.device)
                            _disc_t = torch.tensor([_disc], dtype=torch.long, device=self.device)
                            torch.distributed.all_reduce(_written_t, op=torch.distributed.ReduceOp.SUM)
                            torch.distributed.all_reduce(_disc_t, op=torch.distributed.ReduceOp.SUM)
                            _total_written = int(_written_t.item())
                            _total_disc = int(_disc_t.item())
                        else:
                            _total_written = _written
                            _total_disc = _disc
                        _total_done = _total_written + _total_disc
                        _succ_rate = _total_written / max(_total_done, 1) * 100
                        self._save_replay_buffer(
                            getattr(self, "replay_buffer_path", "")
                        )
                        if rank0_only():
                            ep_dir = self._get_episode_dir()
                            logger.info(
                                f"All episodes exhausted across all ranks. "
                                f"{_total_written} success episodes saved to {ep_dir} "
                                f"(discarded {_total_disc} failed, "
                                f"success rate {_succ_rate:.1f}%)"
                            )
                        self.envs.close()
                        return

                # Check collect_replay_episodes target (synchronised across ranks)
                _collect_ep_target = getattr(self, "collect_replay_episodes", 0)
                if (
                    getattr(self, "collect_replay_only", False)
                    and _collect_ep_target > 0
                ):
                    _written = getattr(self, "_collect_episodes_written", 0)
                    if torch.distributed.is_initialized():
                        _written_t = torch.tensor([_written], dtype=torch.long, device=self.device)
                        torch.distributed.all_reduce(_written_t, op=torch.distributed.ReduceOp.SUM)
                        _total_written = int(_written_t.item())
                    else:
                        _total_written = _written
                    if _total_written >= _collect_ep_target:
                        self._save_replay_buffer(
                            getattr(self, "replay_buffer_path", "")
                        )
                        if rank0_only():
                            ep_dir = self._get_episode_dir()
                            _disc = getattr(self, "_collect_episodes_discarded", 0)
                            if torch.distributed.is_initialized():
                                _disc_t = torch.tensor([_disc], dtype=torch.long, device=self.device)
                                torch.distributed.all_reduce(_disc_t, op=torch.distributed.ReduceOp.SUM)
                                _total_disc = int(_disc_t.item())
                            else:
                                _total_disc = _disc
                            _total_done = _total_written + _total_disc
                            _succ_rate = _total_written / max(_total_done, 1) * 100
                            logger.info(
                                f"Reached collect target: {_total_written}/{_collect_ep_target} "
                                f"success episodes saved to {ep_dir} "
                                f"(discarded {_total_disc} failed, "
                                f"success rate {_succ_rate:.1f}%)"
                            )
                        self.envs.close()
                        return

                # ==================== Social Imagination (reward augmentation) ====================
                imag_stats = {}
                if self.imagine_enabled and self.world_model is not None:
                    # H==1 with cached lookahead: skip expensive _cache_wm_states
                    storage_tmp = self._get_agent0_storage(self._agent.rollouts)
                    has_la_cache = "lookahead_features" in storage_tmp.buffers
                    if self.imagine_horizon > 1 or not has_la_cache:
                        self._cache_wm_states_for_imagination(self._agent.rollouts)
                    imag_stats = self._run_social_imagination(self._agent.rollouts)

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                # ==================== Policy Update ====================
                losses = self._update_agent()

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    losses,
                    count_steps_delta,
                )

                # Log imagination stats
                if self.imagine_enabled and self.world_model is not None:
                    for key, value in imag_stats.items():
                        losses[f'imag_{key}'] = value

                # ==================== World Model Update (if needed) ====================
                if self._should_update_world_model(self.num_updates_done):
                    wm_losses = self._update_world_model(self.num_updates_done)
                    # 添加 WM losses 到总 losses
                    for key, value in wm_losses.items():
                        losses[f'wm_{key}'] = value
                    # 按 wm_vis_interval 存储 WM 可视化到磁盘并写 TensorBoard（wm_vis_interval=0 则不保存）
                    if (
                        self.wm_vis_interval > 0
                        and rank0_only()
                        and (self.num_updates_done % self.wm_vis_interval == 0)
                    ):
                        self._visualize_wm_predictions(
                            writer, self.num_steps_done, num_samples=3
                        )
                        self._visualize_imagination_vs_reality(
                            writer, self.num_steps_done, num_samples=3,
                            horizon=self.imag_loss_horizon,
                        )
                    else:
                        self._last_wm_vis_batch = None

                self._training_log(
                    writer, losses, prev_time,
                    count_checkpoints=count_checkpoints,
                )

                # checkpoint model（resume state 与 ckpt 同频保存）
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    logger.info(f"Saved checkpoint ckpt.{count_checkpoints}.pth")
                    count_checkpoints += 1
                    # 与 ckpt 同频写 resume state，便于断点续训
                    requeue_stats = dict(
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                        run_id=writer.get_run_id(),
                    )
                    save_resume_state(
                        dict(
                            **self._agent.get_resume_state(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                # Periodically save replay buffer to standalone path for offline WM training
                _is_buf_interval = (
                    (self.train_world_model or getattr(self, "collect_replay_only", False))
                    and getattr(self, "replay_buffer_path", "")
                    and getattr(self, "replay_buffer_save_interval", 0) > 0
                    and self.num_updates_done % self.replay_buffer_save_interval == 0
                )
                if _is_buf_interval:
                    if getattr(self, "collect_replay_only", False):
                        self._save_replay_buffer(self.replay_buffer_path)
                    elif rank0_only():
                        self._save_replay_buffer(self.replay_buffer_path)

                profiling_wrapper.range_pop()  # train update

            # Final save of replay buffer when training ends
            _has_buf = (
                (self.train_world_model or getattr(self, "collect_replay_only", False))
                and getattr(self, "replay_buffer_path", "")
            )
            if _has_buf:
                if getattr(self, "collect_replay_only", False):
                    self._save_replay_buffer(self.replay_buffer_path)
                    if rank0_only():
                        ep_dir = self._get_episode_dir()
                        n_eps = getattr(self, "_collect_episodes_written", 0)
                        logger.info(
                            f"Final flush complete. {n_eps} episodes saved to {ep_dir}"
                        )
                elif rank0_only():
                    self._save_replay_buffer(self.replay_buffer_path)
                    logger.info(
                        f"Final replay buffer saved to {self.replay_buffer_path} "
                        f"({self._get_replay_buffer_size()} experiences)"
                    )

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Some configurations require not to load the checkpoint, like when using
        # a hierarchial policy
        if self.config.habitat_baselines.eval.should_load_ckpt:
            # map_location="cpu" is almost always better than mapping to a CUDA device.
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location="cpu"
            )
            step_id = ckpt_dict["extra_state"]["step"]
            logger.info(f"Loaded checkpoint from {checkpoint_path} (trained for {step_id} steps)")
        else:
            ckpt_dict = {"config": None}

        if "config" not in ckpt_dict:
            ckpt_dict["config"] = None

        config = self._get_resume_state_config_or_new_config(
            ckpt_dict["config"]
        )
        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

        if len(self.config.habitat_baselines.eval.video_option) > 0:
            n_agents = len(config.habitat.simulator.agents)
            for agent_i in range(n_agents):
                agent_name = config.habitat.simulator.agents_order[agent_i]
                agent_config = get_agent_config(
                    config.habitat.simulator, agent_i
                )

                agent_sensors = agent_config.sim_sensors
                extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
                with read_write(agent_sensors):
                    agent_sensors.update(extra_sensors)
                with read_write(config):
                    if config.habitat.gym.obs_keys is not None:
                        for render_view in extra_sensors.values():
                            if (
                                render_view.uuid
                                not in config.habitat.gym.obs_keys
                            ):
                                if n_agents > 1:
                                    config.habitat.gym.obs_keys.append(
                                        f"{agent_name}_{render_view.uuid}"
                                    )
                                else:
                                    config.habitat.gym.obs_keys.append(
                                        render_view.uuid
                                    )

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        self._init_envs(config, is_eval=True)

        self._agent = self._create_agent(None)
        if (
            self._agent.actor_critic.should_load_agent_state
            and self.config.habitat_baselines.eval.should_load_ckpt
        ):
            # 兼容单 agent/预训练 ckpt 格式（顶层 state_dict）与多 agent 格式（state[0], state[1]...）
            state_to_load = ckpt_dict
            if "state_dict" in ckpt_dict and 0 not in ckpt_dict and "0" not in ckpt_dict:
                state_to_load = {0: {"state_dict": ckpt_dict["state_dict"]}}
            self._agent.load_state_dict(state_to_load)
            logger.info("Agent state dict loaded (see above for param/key stats)")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        evaluator = hydra.utils.instantiate(config.habitat_baselines.evaluator)
        assert isinstance(evaluator, Evaluator)
        evaluator.evaluate_agent(
            self._agent,
            self.envs,
            config,
            checkpoint_index,
            step_id,
            writer,
            self.device,
            self.obs_transforms,
            self._env_spec,
            self._rank0_keys,
        )

        self.envs.close()


def get_device(config: "DictConfig") -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda", config.habitat_baselines.torch_gpu_id)
        torch.cuda.set_device(device)
        return device
    else:
        return torch.device("cpu")
