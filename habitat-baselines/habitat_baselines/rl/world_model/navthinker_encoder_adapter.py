# Copyright (c) 2024 ForeSightNav-WM
# Adapter to reuse Habitat's ResNetEncoder for World Model
#
# Design Principles:
# 1. 复用 Habitat 代码的部分保持与 Habitat 代码顺序完全一致
# 2. WM 扩展部分清晰标注，不影响 Habitat 原有逻辑
# 3. 代码结构遵循 Habitat ResNetEncoder 的组织方式
#
# Code Structure:
# - __init__:
#   Part 1: Habitat Visual Encoder (复用 ResNetEncoder)
#   Part 2: 1D Sensors Encoder (WM扩展)
#   Part 3: Human Trajectory Encoder (WM扩展)
#   Part 4: Start/Goal Encoder (WM扩展)
#
# - forward:
#   Part 1: Visual Features (复用 Habitat 逻辑 + 全局池化)
#   Part 2: 1D Sensors (WM扩展)
#   Part 3: Human Trajectory (WM扩展)
#   Part 4: Start/Goal (WM扩展)

import logging
import numpy as np
import torch
from torch import nn
import gym.spaces as spaces
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.world_model.networks import (
    VISUAL_KEYS_DEFAULT,
    FUSE_KEYS_1D_DEFAULT,
    HUMAN_STATE_GOAL_KEYS_DEFAULT,
)

logger = logging.getLogger(__name__)


class NavThinkerEncoderAdapter(nn.Module):
    """
    Adapter to reuse Habitat's ResNetEncoder for World Model.

    Converts Habitat's 2D feature maps (C, H, W) to 1D vectors for RSSM
    using global average pooling.

    Features:
    - Reuses Habitat's pretrained ResNet backbone
    - Handles visual observations (depth, rgb)
    - Encodes 1D sensors (GPS, Compass, etc.)
    - Encodes human trajectory (optional)
    - Encodes human start/goal (optional)

    Code structure follows Habitat's ResNetEncoder for consistency.
    """

    def __init__(
        self,
        observation_space,
        baseplanes=32,
        ngroups=32,
        make_backbone=None,
        hidden_size=256,
        use_projection=False,
        target_dim=512,
        visual_keys=None,
        fuse_keys_1d=None,
        human_state_goal_keys=None,
    ):
        """
        Args:
            observation_space: Habitat observation space
            baseplanes: ResNet base planes (32 or 64)
            ngroups: GroupNorm groups (32)
            make_backbone: ResNet factory function (resnet18 or resnet50)
            hidden_size: Hidden size for 1D sensor encoders
            use_projection: Whether to project visual features to target_dim
            target_dim: Target dimension for visual projection
            visual_keys: 3D 视觉观测 key 子串列表，None 时用 VISUAL_KEYS_DEFAULT
            fuse_keys_1d: 1D 观测 key 子串列表，None 时用 FUSE_KEYS_1D_DEFAULT
            human_state_goal_keys: state_goal 2D key 子串，None 时用 HUMAN_STATE_GOAL_KEYS_DEFAULT
        """
        super().__init__()
        _allow_visual = visual_keys if visual_keys is not None else VISUAL_KEYS_DEFAULT
        _allow_1d = fuse_keys_1d if fuse_keys_1d is not None else FUSE_KEYS_1D_DEFAULT
        _allow_hs = human_state_goal_keys if human_state_goal_keys is not None else HUMAN_STATE_GOAL_KEYS_DEFAULT

        # ==================== Part 1: Habitat Visual Encoder (复用Habitat代码) ====================
        # 保持与 Habitat ResNetEncoder

        visual_obs_space = spaces.Dict(
            {
                k: v
                for k, v in observation_space.spaces.items()
                if any(allow in k for allow in _allow_visual)
            }
        )

        self.visual_encoder = ResNetEncoder(
            observation_space=visual_obs_space,
            baseplanes=baseplanes,
            ngroups=ngroups,
            spatial_size=128,
            make_backbone=make_backbone,
            normalize_visual_inputs=False,
        )

        # Calculate visual output dimension (after global pooling)
        visual_outdim = 0
        if not self.visual_encoder.is_blind:
            C, H, W = self.visual_encoder.output_shape  # 遵循Habitat的output_shape命名
            visual_outdim = C  # 全局池化后使用通道维度

            # Optional: project to target dimension (WM扩展)
            if use_projection:
                self.visual_projection = nn.Sequential(
                    nn.Linear(C, target_dim, bias=False),
                    nn.LayerNorm(target_dim, eps=1e-03),
                    nn.SiLU(),
                )
                visual_outdim = target_dim
            else:
                self.visual_projection = None
        else:
            self.visual_projection = None

        # ==================== Part 2: 1D Sensors Encoder (WM扩展) ====================
        # 直接指定要编码的 1D key 子串（与 observation_space 中 key 做子串匹配）
        self._fuse_keys_1d = [
            k for k in observation_space.spaces.keys()
            if len(observation_space.spaces[k].shape) == 1
            and any(allow in k for allow in _allow_1d)
        ]

        sensor_outdim = 0
        if len(self._fuse_keys_1d) > 0:
            fuse_dim = sum(
                observation_space.spaces[k].shape[0]
                for k in self._fuse_keys_1d
            )
            self._fuse_encoder = nn.Sequential(
                nn.Linear(fuse_dim, hidden_size, bias=False),
                nn.LayerNorm(hidden_size, eps=1e-03),
                nn.SiLU(),
            )
            sensor_outdim = hidden_size
        else:
            self._fuse_encoder = None

        # ==================== Part 3: Human state+goal（直接指定 key 子串） ====================
        self._human_state_goal_key = next(
            (k for k in observation_space.spaces.keys()
             if len(observation_space.spaces[k].shape) == 2
             and any(allow in k for allow in _allow_hs)),
            None,
        )
        hs_outdim = 0
        if self._human_state_goal_key is not None:
            hs_shape = observation_space.spaces[self._human_state_goal_key].shape
            hs_flatten_dim = hs_shape[0] * hs_shape[1]
            self._human_state_goal_encoder = nn.Sequential(
                nn.Linear(hs_flatten_dim, hidden_size // 2, bias=False),
                nn.LayerNorm(hidden_size // 2, eps=1e-03),
                nn.SiLU(),
            )
            hs_outdim = hidden_size // 2
        else:
            self._human_state_goal_encoder = None

        # Total output dimension
        self.outdim = visual_outdim + sensor_outdim + hs_outdim

        logger.info("NavThinkerEncoderAdapter initialized:")
        logger.info(f"  - Visual: {visual_outdim} (from Habitat ResNet)")
        logger.info(f"  - 1D Sensors: {sensor_outdim} (pointgoal + human_num)")
        logger.info(f"  - Human state+goal: {hs_outdim}")
        logger.info(f"  - Total: {self.outdim}")

    @property
    def is_blind(self):
        """复用Habitat的is_blind属性"""
        return self.visual_encoder.is_blind

    def forward(self, observations):
        """
        Forward pass - 结构遵循Habitat ResNetEncoder.forward()

        Args:
            observations: Dict with keys like 'depth', 'rgb', 'gps',
                         'human_state_goal', etc.

        Returns:
            1D feature vector: (batch, outdim)

        Order matches Habitat's ResNetEncoder.forward():
        1. Visual processing (Habitat logic)
        2. Additional sensors (WM extension)
        """
        outputs = []

        # ==================== Part 1: Visual Features (复用Habitat逻辑) ====================


        if not self.is_blind:
            ve = self.visual_encoder

            # Replicate ResNetEncoder.forward() but capture intermediate features
            cnn_input = []
            for k in ve.visual_keys:
                obs_k = observations[k]
                obs_k = obs_k.permute(0, 3, 1, 2)
                if ve.key_needs_rescaling[k] is not None:
                    obs_k = obs_k.float() * ve.key_needs_rescaling[k]
                cnn_input.append(obs_k)

            vis = torch.cat(cnn_input, dim=1)
            vis = torch.nn.functional.avg_pool2d(vis, 2)
            vis = ve.running_mean_and_var(vis)

            backbone = ve.backbone
            vis = backbone.conv1(vis)
            vis = backbone.maxpool(vis)
            f1 = backbone.layer1(vis)    # (B, 128, 32, 32)
            f2 = backbone.layer2(f1)     # (B, 256, 16, 16)
            f3 = backbone.layer3(f2)     # (B, 512,  8,  8)
            f4 = backbone.layer4(f3)     # (B, 1024, 4,  4)
            feat_2d = ve.compression(f4) # (B, 128,  4,  4)

            # Global pool → 1D for RSSM
            x = torch.mean(feat_2d, dim=[2, 3])  # (batch, C)

            if self.visual_projection is not None:
                x = self.visual_projection(x)

            outputs.append(x)

        # ==================== Part 2: 1D Sensors (WM扩展) ====================
        if self._fuse_encoder is not None:
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys_1d], dim=-1
            )
            fuse_feats = self._fuse_encoder(fuse_states.float())
            outputs.append(fuse_feats)

        # ==================== Part 3: Human state+goal (agent_0_human_state_goal) ====================
        if self._human_state_goal_encoder is not None and self._human_state_goal_key is not None and self._human_state_goal_key in observations:
            hs_obs = observations[self._human_state_goal_key]
            hs_flat = hs_obs.reshape(hs_obs.shape[0], -1)
            hs_feats = self._human_state_goal_encoder(hs_flat.float())
            outputs.append(hs_feats)

        # ==================== Final: Concatenate All Features ====================
        if len(outputs) == 0:
            return None

        return torch.cat(outputs, dim=-1)

    def load_navthinker_weights(self, checkpoint_path, strict=False):
        """
        Load Habitat's pretrained visual encoder weights.

        Args:
            checkpoint_path: Path to Habitat checkpoint
            strict: Whether to strictly enforce state dict matching
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Extract visual encoder weights
        visual_state_dict = {}
        prefix = 'actor_critic.net.visual_encoder.'

        for k, v in checkpoint['state_dict'].items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                visual_state_dict[new_key] = v

        if len(visual_state_dict) == 0:
            logger.warning(f"No visual encoder weights found in {checkpoint_path}")
            return

        # Load weights
        missing_keys, unexpected_keys = self.visual_encoder.load_state_dict(
            visual_state_dict, strict=strict
        )

        loaded_count = len(visual_state_dict) - len(missing_keys)
        total_params = sum(p.numel() for p in self.visual_encoder.parameters())
        loaded_params = sum(visual_state_dict[k].numel() for k in visual_state_dict if k not in missing_keys)

        if len(missing_keys) > 0:
            logger.warning(f"WM Habitat encoder: missing keys ({len(missing_keys)}): {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"WM Habitat encoder: unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")

        logger.info(
            f"[World Model] Successfully loaded pretrained Habitat encoder weights!\n"
            f"  Source: {checkpoint_path}\n"
            f"  Loaded layers: {loaded_count}/{len(visual_state_dict)}\n"
            f"  Loaded params: {loaded_params:,} / {total_params:,} ({loaded_params/total_params*100:.1f}%)"
        )
