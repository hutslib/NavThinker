#!/usr/bin/env python3

"""
NavThinker + DINO-WM Lookahead Policy

Architecture:
  - DA-V2 (frozen ViT) as policy visual encoder
  - DINO-WM causal ViT predictor as world model
  - Lookahead planning: imagine 1-step for each action, late-fuse with RNN output
  - NavThinker RNN backbone unchanged
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, TYPE_CHECKING
from gym import spaces
import numpy as np

from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ddppo.policy.resnet_policy import (
    ResNetEncoder,
    ResNetCLIPEncoder,
)
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.world_model.models import SocialNavWorldModel
from habitat_baselines.rl.world_model import tools
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalSensor
from habitat.tasks.nav.nav import (
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ProximitySensor,
    EpisodicCompassSensor,
    ImageGoalSensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor

if TYPE_CHECKING:
    from omegaconf import DictConfig


class DAV2PolicyEncoder(nn.Module):
    """Policy visual encoder: frozen DA-V2 (DINOv2 ViT) backbone.

    Patch features are mean-pooled into a single (B, vit_embed_dim) vector.
    """

    def __init__(self, observation_space, da_config):
        super().__init__()
        from habitat_baselines.rl.world_model.depth_anything_adapter import (
            DepthAnythingEncoderAdapter,
        )
        self.da_encoder = DepthAnythingEncoderAdapter(
            observation_space=observation_space,
            vit_model_name=da_config.get('da_vit_model', 'vits'),
            vit_pretrained_path=da_config.get('da_pretrained_path', None),
            hidden_size=da_config.get('mlp_units', 256),
            projection_dim=da_config.get('da_projection_dim', 512),
            vit_input_size=da_config.get('da_vit_input_size', 518),
            visual_keys=da_config.get('visual_keys'),
            fuse_keys_1d=da_config.get('fuse_keys_1d'),
            human_state_goal_keys=da_config.get('human_state_goal_keys'),
        )
        self._output_shape = (self.da_encoder.vit_embed_dim,)

    @property
    def is_blind(self):
        return self.da_encoder.is_blind

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, observations):
        x = self.da_encoder._preprocess_rgb(observations)
        patch_features, cls_token = self.da_encoder._extract_vit_features(x)
        return patch_features.mean(dim=1)


class NavThinkerNet(Net):
    """NavThinker RNN backbone + DINO-WM lookahead late fusion.

    Data flow:
      1. DA-V2 encoder → visual_fc → visual_feats (B, hidden_size)
      2. visual_feats + 1D sensors + prev_action → RNN → rnn_out (B, hidden_size)
      3. DINO-WM lookahead: for each action, img_step_lookahead → per-action FC
         → concat all actions → lookahead_feat (B, lookahead_dim * A)
      4. Late fusion: [LayerNorm(rnn_out), WM_LN(wm_feat), lookahead_feat] → actor/critic
    """

    PRETRAINED_VISUAL_FEATURES_KEY = "visual_features"
    LOOKAHEAD_CACHED_KEY = "lookahead_cached_feature"
    prev_action_embedding: nn.Module

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
        fuse_keys: Optional[List[str]],
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        world_model: Optional[SocialNavWorldModel] = None,
        da_encoder_config=None,
        lookahead_dim: int = 128,
        policy_encoder_type: str = "dav2",
    ):
        super().__init__()

        self.discrete_actions = discrete_actions
        self._n_prev_action = 32
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(
                action_space.n + 1, self._n_prev_action
            )
        else:
            from habitat_baselines.utils.common import get_num_actions
            num_actions = get_num_actions(action_space)
            self.prev_action_embedding = nn.Linear(
                num_actions, self._n_prev_action
            )
        rnn_input_size = self._n_prev_action

        # 1D state sensor fusion
        if fuse_keys is None:
            fuse_keys = observation_space.spaces.keys()
            goal_sensor_keys = {
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid,
                ObjectGoalSensor.cls_uuid,
                EpisodicGPSSensor.cls_uuid,
                PointGoalSensor.cls_uuid,
                HeadingSensor.cls_uuid,
                ProximitySensor.cls_uuid,
                EpisodicCompassSensor.cls_uuid,
                ImageGoalSensor.cls_uuid,
                InstanceImageGoalSensor.cls_uuid,
            }
            fuse_keys = [k for k in fuse_keys if k not in goal_sensor_keys]
        self._fuse_keys_1d: List[str] = [
            k for k in fuse_keys
            if len(observation_space.spaces[k].shape) == 1
            and k != "human_num_sensor" and k != "localization_sensor"
        ]
        if len(self._fuse_keys_1d) != 0:
            rnn_input_size += sum(
                observation_space.spaces[k].shape[0]
                for k in self._fuse_keys_1d
            )

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observation_space.spaces:
            n_input_goal = (
                observation_space.spaces[
                    IntegratedPointGoalGPSAndCompassSensor.cls_uuid
                ].shape[0] + 1
            )
            self.tgt_embeding = nn.Linear(n_input_goal, 32)
            rnn_input_size += 32

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]) + 1
            )
            self.obj_categories_embedding = nn.Embedding(self._n_object_categories, 32)
            rnn_input_size += 32

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[EpisodicGPSSensor.cls_uuid].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32

        if PointGoalSensor.cls_uuid in observation_space.spaces:
            input_pointgoal_dim = observation_space.spaces[PointGoalSensor.cls_uuid].shape[0]
            self.pointgoal_embedding = nn.Linear(input_pointgoal_dim, 32)
            rnn_input_size += 32

        if HeadingSensor.cls_uuid in observation_space.spaces:
            input_heading_dim = observation_space.spaces[HeadingSensor.cls_uuid].shape[0] + 1
            assert input_heading_dim == 2, "Expected heading with 2D rotation."
            self.heading_embedding = nn.Linear(input_heading_dim, 32)
            rnn_input_size += 32

        if ProximitySensor.cls_uuid in observation_space.spaces:
            input_proximity_dim = observation_space.spaces[ProximitySensor.cls_uuid].shape[0]
            self.proximity_embedding = nn.Linear(input_proximity_dim, 32)
            rnn_input_size += 32

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[0] == 1
            ), "Expected compass with 2D rotation."
            self.compass_embedding = nn.Linear(2, 32)
            rnn_input_size += 32

        for uuid in [ImageGoalSensor.cls_uuid, InstanceImageGoalSensor.cls_uuid]:
            if uuid in observation_space.spaces:
                goal_observation_space = spaces.Dict(
                    {"rgb": observation_space.spaces[uuid]}
                )
                goal_visual_encoder = ResNetEncoder(
                    goal_observation_space,
                    baseplanes=resnet_baseplanes,
                    ngroups=resnet_baseplanes // 2,
                    make_backbone=getattr(resnet, backbone),
                    normalize_visual_inputs=normalize_visual_inputs,
                )
                setattr(self, f"{uuid}_encoder", goal_visual_encoder)
                goal_visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod(goal_visual_encoder.output_shape), hidden_size),
                    nn.ReLU(True),
                )
                setattr(self, f"{uuid}_fc", goal_visual_fc)
                rnn_input_size += hidden_size

        self._hidden_size = hidden_size
        self._policy_encoder_type = policy_encoder_type

        # ==================== Policy Visual Encoder ====================
        if policy_encoder_type == "resnet":
            if force_blind_policy:
                use_obs_space = spaces.Dict({})
            else:
                use_obs_space = spaces.Dict(
                    {
                        k: observation_space.spaces[k]
                        for k in (fuse_keys if fuse_keys else [])
                        if k in observation_space.spaces
                        and len(observation_space.spaces[k].shape) == 3
                        and "rgb" not in k.lower()
                    }
                )
            self.visual_encoder = ResNetEncoder(
                use_obs_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs,
            )
            if not self.visual_encoder.is_blind:
                self.visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(self.visual_encoder.output_shape), hidden_size
                    ),
                    nn.ReLU(True),
                )
        else:
            if da_encoder_config is None:
                raise ValueError("da_encoder_config is required for dav2 policy encoder")
            self.visual_encoder = DAV2PolicyEncoder(observation_space, da_encoder_config)
            if not self.visual_encoder.is_blind:
                self.visual_fc = nn.Sequential(
                    nn.Linear(self.visual_encoder.output_shape[0], hidden_size),
                    nn.ReLU(True),
                )

        # ==================== World Model (DINO-WM) ====================
        self.world_model = world_model
        self.use_world_model = world_model is not None
        self.rssm_state = None

        wm_feat_size = world_model.dynamics.feat_size if world_model is not None else 0

        # ==================== Lookahead (late fusion) ====================
        num_actions = action_space.n if hasattr(action_space, "n") else 4
        self._lookahead_num_actions = num_actions
        self._lookahead_dim = lookahead_dim
        self._lookahead_late_dim = 0

        if self.use_world_model:
            self.lookahead_per_action_fc = nn.Sequential(
                nn.Linear(wm_feat_size, lookahead_dim, bias=False),
                nn.LayerNorm(lookahead_dim, eps=1e-03),
                nn.SiLU(),
            )
            self.lookahead_per_action_fc.apply(tools.weight_init)
            self._lookahead_late_dim = lookahead_dim * num_actions
            self.rnn_out_ln = nn.LayerNorm(hidden_size, eps=1e-03)

        # ==================== RNN ====================
        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )
        self.train()

    @property
    def output_size(self):
        base = self._hidden_size
        if self._lookahead_late_dim > 0:
            base += self._lookahead_late_dim
        return base

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    def _encode_lookahead_late(self, la_concat, batch_size):
        """Per-action encode → (B, lookahead_dim * A)."""
        num_a = self._lookahead_num_actions
        feat_d = la_concat.shape[-1] // num_a
        per_action = la_concat.view(batch_size, num_a, feat_d)
        encoded = self.lookahead_per_action_fc(per_action)
        return encoded.view(batch_size, -1)

    def _run_lookahead(self, rssm_state, batch_size, device):
        """Imagine 1-step for each action, return (B, D*A) raw features."""
        la_feats = []
        with torch.no_grad():
            for a_idx in range(self._lookahead_num_actions):
                act_oh = torch.zeros(
                    batch_size, self._lookahead_num_actions, device=device,
                )
                act_oh[:, a_idx] = 1.0
                pred_state = self.world_model.dynamics.img_step_lookahead(
                    rssm_state, act_oh
                )
                la_feats.append(pred_state["deter"].mean(dim=-2))
        return torch.cat(la_feats, dim=-1)

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = []
        aux_loss_state = {}
        lookahead_feat_for_late = None

        if not self.is_blind:
            if self.PRETRAINED_VISUAL_FEATURES_KEY in observations:  # noqa: SIM401
                visual_feats = observations[self.PRETRAINED_VISUAL_FEATURES_KEY]
            else:
                visual_feats = self.visual_encoder(observations)
            visual_feats = self.visual_fc(visual_feats)
            aux_loss_state["perception_embed"] = visual_feats

            if self.use_world_model:
                batch_size = prev_actions.shape[0]
                cached_la = observations.get(self.LOOKAHEAD_CACHED_KEY)
                use_persistent = rnn_build_seq_info is None
                use_cached = (
                    cached_la is not None
                    and (not use_persistent or not self.training)
                )

                if use_persistent:
                    dyn = self.world_model.dynamics
                    need_reset = (
                        self.rssm_state is None
                        or self.rssm_state["deter"].shape[0] != batch_size
                    )
                    if need_reset:
                        self.rssm_state = dyn.initial(batch_size)
                        dyn.reset_imagination()
                        if hasattr(dyn, 'reset_kv_cache'):
                            dyn.reset_kv_cache(batch_size)
                    rssm_state = self.rssm_state
                else:
                    rssm_state = self.world_model.dynamics.initial(batch_size)
                    if hasattr(self.world_model.dynamics, 'reset_kv_cache'):
                        self.world_model.dynamics.reset_kv_cache(batch_size)

                if use_cached:
                    # PPO update path: use rollout-cached features
                    lookahead_feat_for_late = self._encode_lookahead_late(
                        cached_la, batch_size
                    )
                else:
                    # Rollout path: run WM encoder + lookahead
                    with torch.no_grad():
                        wm_embed = self.world_model.encoder(observations)
                        wm_embed = getattr(
                            self.world_model.encoder,
                            '_cached_vit_patch_features', wm_embed
                        )

                    if self.discrete_actions:
                        action_one_hot = torch.zeros(
                            batch_size,
                            self.prev_action_embedding.num_embeddings - 1,
                            device=prev_actions.device,
                        )
                        valid_actions = (prev_actions >= 0).squeeze(-1)
                        if valid_actions.any():
                            action_one_hot[valid_actions] = F.one_hot(
                                prev_actions[valid_actions].squeeze(-1),
                                num_classes=action_one_hot.shape[1],
                            ).float()
                    else:
                        action_one_hot = prev_actions

                    with torch.no_grad():
                        embed_seq = wm_embed.unsqueeze(1)
                        is_first = (~masks).float()

                        if use_persistent:
                            # Scheme B+: defer writing obs_t into _hist_z.
                            # act() will call observe_post_action after action selection.
                            dyn = self.world_model.dynamics
                            is_first_flat = is_first.view(batch_size).bool()
                            if is_first_flat.any() and dyn._hist_z is not None:
                                reset_mask = is_first_flat.view(batch_size, 1, 1, 1)
                                dyn._hist_z = dyn._hist_z * (~reset_mask).float()

                            rssm_state = {"deter": embed_seq[:, 0]}
                            wm_feat = None
                        else:
                            rssm_state = self.world_model.dynamics.initial(batch_size)

                    # Lookahead
                    if use_persistent:
                        cached_la = observations.get(self.LOOKAHEAD_CACHED_KEY)
                        if cached_la is not None:
                            la_concat = cached_la
                        else:
                            la_concat = self._run_lookahead(
                                rssm_state, batch_size, embed_seq.device
                            )
                            aux_loss_state["lookahead_feature_for_storage"] = la_concat.detach()

                        lookahead_feat_for_late = self._encode_lookahead_late(
                            la_concat.detach(), batch_size
                        )

                        aux_loss_state["_la_embed_seq"] = embed_seq.detach()
                        aux_loss_state["_la_is_first"] = is_first.detach()

                if use_persistent:
                    self.rssm_state = rssm_state

            x.append(visual_feats)

        # 1D state sensors
        if len(self._fuse_keys_1d) != 0:
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys_1d], dim=-1
            )
            x.append(fuse_states.float())

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            if goal_observations.shape[1] == 2:
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
            else:
                assert goal_observations.shape[1] == 3, "Unsupported dimensionality"
                vertical_angle_sin = torch.sin(goal_observations[:, 2])
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]) * vertical_angle_sin,
                        torch.sin(-goal_observations[:, 1]) * vertical_angle_sin,
                        torch.cos(goal_observations[:, 2]),
                    ],
                    -1,
                )
            x.append(self.tgt_embeding(goal_observations))

        if PointGoalSensor.cls_uuid in observations:
            x.append(self.pointgoal_embedding(observations[PointGoalSensor.cls_uuid]))

        if ProximitySensor.cls_uuid in observations:
            x.append(self.proximity_embedding(observations[ProximitySensor.cls_uuid]))

        if HeadingSensor.cls_uuid in observations:
            sensor_observations = observations[HeadingSensor.cls_uuid]
            sensor_observations = torch.stack(
                [torch.cos(sensor_observations[0]), torch.sin(sensor_observations[0])],
                -1,
            )
            x.append(self.heading_embedding(sensor_observations))

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(self.compass_embedding(compass_observations.squeeze(dim=1)))

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid]))

        for uuid in [ImageGoalSensor.cls_uuid, InstanceImageGoalSensor.cls_uuid]:
            if uuid in observations:
                goal_image = observations[uuid]
                goal_visual_encoder = getattr(self, f"{uuid}_encoder")
                goal_visual_output = goal_visual_encoder({"rgb": goal_image})
                goal_visual_fc = getattr(self, f"{uuid}_fc")
                x.append(goal_visual_fc(goal_visual_output))

        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        else:
            prev_actions = self.prev_action_embedding(masks * prev_actions.float())

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )

        # Late fusion: LN(rnn_out) || lookahead_feat
        if lookahead_feat_for_late is not None:
            out = torch.cat(
                [self.rnn_out_ln(out), lookahead_feat_for_late], dim=-1
            )

        aux_loss_state["rnn_output"] = out
        return out, rnn_hidden_states, aux_loss_state

    def on_envs_pause(self, envs_to_pause):
        if not envs_to_pause:
            return
        if self.rssm_state is not None:
            keep = [i for i in range(self.rssm_state["deter"].shape[0])
                    if i not in envs_to_pause]
            if len(keep) == 0:
                self.rssm_state = None
            else:
                self.rssm_state = {
                    k: v[keep] for k, v in self.rssm_state.items()
                }
        if self.use_world_model:
            self.world_model.dynamics.on_envs_pause(envs_to_pause)


@baseline_registry.register_policy
class NavThinkerPolicy(NetPolicy):
    """NavThinker + DINO-WM Lookahead Policy."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        fuse_keys: Optional[List[str]] = None,
        use_world_model: bool = False,
        world_model_config=None,
        wm_fusion_mode: str = "late",
        policy_encoder_type: str = "dav2",
        policy_encoder_config: Optional[Dict] = None,
        lookahead_enabled: bool = False,
        lookahead_steps: int = 1,
        lookahead_dim: int = 128,
        **kwargs,
    ):
        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
            self.action_distribution_type = policy_config.action_distribution_type
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"

        world_model = None
        if use_world_model and world_model_config is not None:
            world_model = SocialNavWorldModel(
                world_model_config,
                observation_space=observation_space,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

        net = NavThinkerNet(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            rnn_type=rnn_type,
            backbone=backbone,
            resnet_baseplanes=resnet_baseplanes,
            normalize_visual_inputs=normalize_visual_inputs,
            fuse_keys=fuse_keys,
            force_blind_policy=force_blind_policy,
            discrete_actions=discrete_actions,
            world_model=world_model,
            da_encoder_config=policy_encoder_config,
            lookahead_dim=lookahead_dim,
            policy_encoder_type=policy_encoder_type,
        )

        super().__init__(
            net,
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

    def on_envs_pause(self, envs_to_pause):
        if hasattr(self.net, "on_envs_pause"):
            self.net.on_envs_pause(envs_to_pause)

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        from collections import OrderedDict

        agent_name = kwargs.get("agent_name")
        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            agent_name = config.habitat.simulator.agents_order[0]

        ignore_names = [
            sensor.uuid
            for sensor in config.habitat_baselines.eval.extra_sim_sensors.values()
        ]
        filtered_obs = spaces.Dict(
            OrderedDict(
                (k, v) for k, v in observation_space.items()
                if k not in ignore_names
            )
        )

        # World Model config
        use_world_model = False
        world_model_config = None
        wm_cfg = getattr(config.habitat_baselines, "world_model", None)
        if wm_cfg is not None:
            use_world_model = getattr(wm_cfg, "enabled", False)
            if use_world_model:
                num_actions = action_space.n if hasattr(action_space, "n") else 4
                wm_config_dict = {
                    "encoder_type": getattr(wm_cfg, "encoder_type", "depth_anything"),
                    "dynamics_type": getattr(wm_cfg, "dynamics_type", "dino_wm"),
                    "dyn_stoch": getattr(wm_cfg, "dyn_stoch", 30),
                    "dyn_deter": getattr(wm_cfg, "dyn_deter", 200),
                    "dyn_hidden": getattr(wm_cfg, "dyn_hidden", 200),
                    "dyn_discrete": getattr(wm_cfg, "dyn_discrete", 32),
                    "cnn_depth": getattr(wm_cfg, "cnn_depth", 32),
                    "mlp_units": getattr(wm_cfg, "mlp_units", 256),
                    "kernel_size": getattr(wm_cfg, "kernel_size", 4),
                    "minres": getattr(wm_cfg, "minres", 4),
                    "act": getattr(wm_cfg, "act", "SiLU"),
                    "norm": getattr(wm_cfg, "norm", True),
                    "num_actions": num_actions,
                    "visual_keys": getattr(wm_cfg, "visual_keys", None),
                    "fuse_keys_1d": getattr(wm_cfg, "fuse_keys_1d", None),
                    "human_state_goal_keys": getattr(wm_cfg, "human_state_goal_keys", None),
                    "num_humans": getattr(wm_cfg, "num_humans", 6),
                    "pred_horizon": getattr(wm_cfg, "pred_horizon", 5),
                    "use_goal_conditioning": getattr(wm_cfg, "use_goal_conditioning", True),
                    "state_goal_dim": getattr(wm_cfg, "state_goal_dim", 8),
                    "da_vit_model": getattr(wm_cfg, "da_vit_model", "vits"),
                    "da_pretrained_path": getattr(wm_cfg, "da_pretrained_path", None),
                    "da_projection_dim": getattr(wm_cfg, "da_projection_dim", 512),
                    "da_vit_input_size": getattr(wm_cfg, "da_vit_input_size", 518),
                    "feat_decoder_hidden": getattr(wm_cfg, "feat_decoder_hidden", 512),
                    "feat_decoder_layers": getattr(wm_cfg, "feat_decoder_layers", 2),
                    "dpt_head_features": getattr(wm_cfg, "dpt_head_features", 64),
                    "dino_wm_depth": getattr(wm_cfg, "dino_wm_depth", 6),
                    "dino_wm_heads": getattr(wm_cfg, "dino_wm_heads", 16),
                    "dino_wm_mlp_dim": getattr(wm_cfg, "dino_wm_mlp_dim", 2048),
                    "dino_wm_dim_head": getattr(wm_cfg, "dino_wm_dim_head", 64),
                    "dino_wm_num_hist": getattr(wm_cfg, "dino_wm_num_hist", 3),
                    "dino_wm_dropout": getattr(wm_cfg, "dino_wm_dropout", 0.1),
                    "dino_wm_emb_dropout": getattr(wm_cfg, "dino_wm_emb_dropout", 0.0),
                }

                class WMConfig:
                    def __init__(self, config_dict):
                        self._config = config_dict
                    def get(self, key, default=None):
                        return self._config.get(key, default)

                world_model_config = WMConfig(wm_config_dict)

        # Policy encoder config
        policy_encoder_config = None
        pe_cfg = getattr(config.habitat_baselines, "policy_encoder", None)
        if pe_cfg is not None:
            policy_encoder_config = {
                "da_vit_model": getattr(pe_cfg, "da_vit_model", "vits"),
                "da_pretrained_path": getattr(pe_cfg, "da_pretrained_path", None),
                "da_projection_dim": getattr(pe_cfg, "da_projection_dim", 512),
                "da_vit_input_size": getattr(pe_cfg, "da_vit_input_size", 518),
                "mlp_units": getattr(pe_cfg, "mlp_units", 256),
                "visual_keys": getattr(pe_cfg, "visual_keys", None),
                "fuse_keys_1d": getattr(pe_cfg, "fuse_keys_1d", None),
                "human_state_goal_keys": getattr(pe_cfg, "human_state_goal_keys", None),
            }

        lookahead_dim = 128
        if wm_cfg is not None:
            lookahead_dim = getattr(wm_cfg, "lookahead_dim", 128)

        policy_encoder_type = "dav2"
        if pe_cfg is not None:
            policy_encoder_type = getattr(pe_cfg, "type", "dav2")

        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            num_recurrent_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            resnet_baseplanes=config.habitat_baselines.rl.ddppo.resnet_baseplanes,
            backbone=config.habitat_baselines.rl.ddppo.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.habitat_baselines.force_blind_policy,
            policy_config=config.habitat_baselines.rl.policy[agent_name],
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            fuse_keys=None,
            use_world_model=use_world_model,
            world_model_config=world_model_config,
            policy_encoder_type=policy_encoder_type,
            policy_encoder_config=policy_encoder_config,
            lookahead_dim=lookahead_dim,
        )
