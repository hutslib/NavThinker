# Copyright (c) 2024 ForeSightNav-WM
# Social Navigation World Model Implementation (V2)
#
# This is the V2 version designed for NavThinker policy enhancement.
# Key differences from V1:
# - No standalone train_step() - training handled by navthinker_trainer.py
# - No internal optimizer - optimizer created externally in trainer
# - Only provides core components: encoder, dynamics (RSSM), and heads
# - Supports imagination rollouts for social-aware latent planning
#
# Encoder options:
# - 'dreamer': Dreamer-style CNN (default, designed for WM)
# - 'navthinker': Reuse NavThinker's ResNet encoder (can use pretrained weights)
# - 'depth_anything': Frozen DINOv2 ViT (Depth Anything V2)
#
# Dynamics options:
# - 'gru': DreamerV3-style GRU-based RSSM
# - 'dino_wm': DINO-WM Causal ViT Predictor (deterministic, patch-level)

import logging
import numpy as np
import torch
from torch import nn
from habitat_baselines.rl.world_model.networks import (
    RSSM,
    SocialNavEncoder,
    DepthDecoder,
    DPTDepthHead,
    ViTFeatureDecoder,
    HumanTrajectoryDecoder,
    RewardDecoder,
)
from habitat_baselines.rl.world_model.dino_wm_predictor import DinoWMDynamics
from habitat_baselines.rl.world_model.navthinker_encoder_adapter import NavThinkerEncoderAdapter
from habitat_baselines.rl.ddppo.policy.resnet import resnet18, resnet50

logger = logging.getLogger(__name__)


class SocialNavWorldModel(nn.Module):
    """
    Human-Forecasting Latent World Model for Social Navigation (V2)

    Provides core World Model components for NavThinker policy enhancement:
    - Encoder: Observations → Embeddings
    - RSSM Dynamics: Latent state transitions
    - Decoders: Depth reconstruction, Human trajectory prediction, Reward prediction

    Training is managed externally by navthinker_trainer.py which directly
    calls these components for maximum flexibility.
    """
    def __init__(self, config, observation_space=None, device="cuda"):
        super(SocialNavWorldModel, self).__init__()
        self._config = config
        self.device = device

        encoder_type = config.get('encoder_type', 'dreamer')
        dynamics_type = config.get('dynamics_type', 'gru')

        # ==================== Encoder ====================
        # DINO-WM uses raw ViT patch features directly (no projection/concat),
        # so encoder runs in global mode; raw patches come from the cache.
        is_dino_wm = (dynamics_type == 'dino_wm')

        if encoder_type == 'depth_anything':
            logger.info("[World Model] Using Depth Anything V2 (frozen DINOv2 ViT) encoder")
            if observation_space is None:
                raise ValueError("observation_space is required for Depth Anything encoder")

            from habitat_baselines.rl.world_model.depth_anything_adapter import DepthAnythingEncoderAdapter
            self.encoder = DepthAnythingEncoderAdapter(
                observation_space=observation_space,
                vit_model_name=config.get('da_vit_model', 'vits'),
                vit_pretrained_path=config.get('da_pretrained_path', None),
                hidden_size=config.get('mlp_units', 256),
                projection_dim=config.get('da_projection_dim', 512),
                vit_input_size=config.get('da_vit_input_size', 518),
                visual_keys=config.get('visual_keys'),
                fuse_keys_1d=config.get('fuse_keys_1d'),
                human_state_goal_keys=config.get('human_state_goal_keys'),
            )

        elif encoder_type == 'navthinker':
            logger.info("[World Model] Using NavThinker ResNet encoder (can reuse pretrained weights)")
            if observation_space is None:
                raise ValueError("observation_space is required for NavThinker encoder")

            backbone_name = config.get('navthinker_backbone', 'resnet18')
            make_backbone = resnet18 if backbone_name == 'resnet18' else resnet50

            self.encoder = NavThinkerEncoderAdapter(
                observation_space=observation_space,
                baseplanes=config.get('navthinker_baseplanes', 32),
                ngroups=config.get('navthinker_ngroups', 32),
                make_backbone=make_backbone,
                hidden_size=config.get('mlp_units', 256),
                use_projection=config.get('navthinker_use_projection', False),
                target_dim=config.get('navthinker_target_dim', 512),
                visual_keys=config.get('visual_keys'),
                fuse_keys_1d=config.get('fuse_keys_1d'),
                human_state_goal_keys=config.get('human_state_goal_keys'),
            )

            pretrained_path = config.get('navthinker_pretrained_path', None)
            if pretrained_path:
                logger.info(f"[World Model] Loading pretrained NavThinker encoder from: {pretrained_path}")
                self.encoder.load_navthinker_weights(
                    pretrained_path,
                    strict=config.get('navthinker_strict_load', False)
                )
            else:
                logger.info("[World Model] No pretrained NavThinker encoder path specified, training from scratch")

        elif encoder_type == 'dreamer':
            logger.info("[World Model] Using Dreamer-style CNN encoder")
            if observation_space is not None:
                self.encoder = SocialNavEncoder(
                    observation_space=observation_space,
                    depth=config.get('cnn_depth', 32),
                    act=config.get('act', 'SiLU'),
                    norm=config.get('norm', True),
                    kernel_size=config.get('kernel_size', 4),
                    minres=config.get('minres', 4),
                    hidden_size=config.get('mlp_units', 256),
                    visual_keys=config.get('visual_keys'),
                    fuse_keys_1d=config.get('fuse_keys_1d'),
                    human_state_goal_keys=config.get('human_state_goal_keys'),
                )
            else:
                from gym import spaces
                dummy_obs_space = spaces.Dict({
                    'depth': spaces.Box(
                        low=0, high=255,
                        shape=config.get('depth_shape', (256, 256, 1)),
                        dtype=np.uint8
                    )
                })
                self.encoder = SocialNavEncoder(
                    observation_space=dummy_obs_space,
                    depth=config.get('cnn_depth', 32),
                    act=config.get('act', 'SiLU'),
                    norm=config.get('norm', True),
                    kernel_size=config.get('kernel_size', 4),
                    minres=config.get('minres', 4),
                    hidden_size=config.get('mlp_units', 256),
                )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Must be 'dreamer', 'navthinker', or 'depth_anything'")

        self.embed_size = self.encoder.outdim

        # ==================== Dynamics ====================
        dyn_stoch = config.get('dyn_stoch', 30)
        dyn_deter = config.get('dyn_deter', 200)
        dyn_discrete = config.get('dyn_discrete', 32)

        if dynamics_type == 'dino_wm':
            vit_dim = self.encoder.vit_embed_dim if hasattr(self.encoder, 'vit_embed_dim') else 384
            n_patches_vit = self.encoder.vit_num_patches if hasattr(self.encoder, 'vit_num_patches') else 1369
            logger.info(
                f"[World Model] Using DINO-WM Causal ViT Predictor: "
                f"{n_patches_vit} patches, dim={vit_dim}"
            )
            self.dynamics = DinoWMDynamics(
                dim=vit_dim,
                num_patches=n_patches_vit,
                num_hist=config.get('dino_wm_num_hist', 3),
                num_pred=1,
                action_dim=config.get('num_actions', 4),
                depth=config.get('dino_wm_depth', 6),
                heads=config.get('dino_wm_heads', 16),
                mlp_dim=config.get('dino_wm_mlp_dim', 2048),
                dim_head=config.get('dino_wm_dim_head', 64),
                dropout=config.get('dino_wm_dropout', 0.1),
                emb_dropout=config.get('dino_wm_emb_dropout', 0.0),
                device=device,
            )
            dyn_deter = vit_dim
            dyn_discrete = 0
            feat_size = vit_dim

        else:
            logger.info("[World Model] Using GRU-based RSSM dynamics")
            self.dynamics = RSSM(
                stoch=dyn_stoch,
                deter=dyn_deter,
                hidden=config.get('dyn_hidden', dyn_deter),
                rec_depth=config.get('dyn_rec_depth', 1),
                discrete=dyn_discrete,
                act=config.get('act', 'SiLU'),
                norm=config.get('norm', True),
                mean_act=config.get('dyn_mean_act', 'none'),
                std_act=config.get('dyn_std_act', 'softplus'),
                min_std=config.get('dyn_min_std', 0.1),
                unimix_ratio=config.get('unimix_ratio', 0.01),
                initial=config.get('initial', 'learned'),
                num_actions=config.get('num_actions', 4),
                embed_size=self.embed_size,
                device=device,
            )

        # Feature size (stoch + deter) -- 1D pooled size for reward/traj decoders
        if not is_dino_wm:
            if dyn_discrete:
                feat_size = dyn_stoch * dyn_discrete + dyn_deter
            else:
                feat_size = dyn_stoch + dyn_deter
        self._is_patch_dynamics = is_dino_wm
        self._is_dino_wm = is_dino_wm

        # ==================== Decoders ====================
        self.heads = nn.ModuleDict()

        if is_dino_wm:
            patch_feat_dim = self.dynamics.patch_feat_size
            n_patches_vit = self.encoder.vit_num_patches
            patch_spatial_h = int(n_patches_vit ** 0.5)
            patch_spatial_w = patch_spatial_h

            self.heads["depth"] = DPTDepthHead(
                patch_feat_dim=patch_feat_dim,
                num_patches_h=patch_spatial_h,
                num_patches_w=patch_spatial_w,
                depth_shape=config.get('depth_shape', (256, 256, 1)),
                features=config.get('dpt_head_features', 64),
                act=config.get('act', 'SiLU'),
                norm=config.get('norm', True),
            )
            logger.info(
                f"[World Model] DPT depth head: patch_feat({patch_feat_dim}) "
                f"@ {patch_spatial_h}x{patch_spatial_w}"
            )
        else:
            # Standard 1D dynamics: use ConvTranspose depth decoder
            self.heads["depth"] = DepthDecoder(
                feat_size=feat_size,
                depth_shape=config.get('depth_shape', (256, 256, 1)),
                depth=config.get('decoder_depth', 32),
                act=config.get('act', 'SiLU'),
                norm=config.get('norm', True),
                kernel_size=config.get('kernel_size', 4),
                minres=config.get('minres', 4),
                outscale=config.get('decoder_outscale', 1.0),
            )

            # ViT feature decoder (only for depth_anything encoder with 1D dynamics)
            if encoder_type == 'depth_anything' and not self.encoder.is_blind:
                vit_dim = self.encoder.vit_embed_dim
                n_patches = self.encoder.vit_num_patches
                self.heads["vit_features"] = ViTFeatureDecoder(
                    feat_size=feat_size,
                    vit_embed_dim=vit_dim,
                    num_patches=n_patches,
                    hidden_size=config.get('feat_decoder_hidden', 512),
                    num_layers=config.get('feat_decoder_layers', 2),
                    act=config.get('act', 'SiLU'),
                    norm=config.get('norm', True),
                )
                logger.info(
                    f"[World Model] ViT feature decoder: feat→{n_patches}×{vit_dim}"
                )

        # Human trajectory decoder
        self.heads["human_traj"] = HumanTrajectoryDecoder(
            feat_size=feat_size,
            num_humans=config.get('num_humans', 10),
            pred_horizon=config.get('pred_horizon', 12),
            traj_dim=config.get('traj_dim', 2),
            hidden_layers=config.get('traj_hidden_layers', 3),
            hidden_units=config.get('traj_hidden_units', 256),
            act=config.get('act', 'SiLU'),
            norm=config.get('norm', True),
            outscale=config.get('decoder_outscale', 1.0),
            use_goal_conditioning=config.get('use_goal_conditioning', True),
            state_goal_dim=config.get('state_goal_dim', 8),
            residual=config.get('residual', True),
        )

        # Reward decoder
        self.heads["reward"] = RewardDecoder(
            feat_size=feat_size,
            hidden_size=config.get('reward_hidden', 256),
            act=config.get('act', 'SiLU'),
            norm=config.get('norm', True),
            outscale=config.get('decoder_outscale', 1.0),
        )

    def freeze_encoder(self):
        """冻结 encoder 参数，仅训练 RSSM + decoders"""
        frozen_count = 0
        for param in self.encoder.parameters():
            param.requires_grad_(False)
            frozen_count += param.numel()
        total_count = sum(p.numel() for p in self.parameters())
        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"[World Model] Froze WM encoder: "
            f"{frozen_count:,} params frozen, "
            f"{trainable_count:,} params trainable, "
            f"{total_count:,} total"
        )

    def trainable_parameters(self):
        """返回需要训练的参数（排除 frozen encoder）"""
        return (p for p in self.parameters() if p.requires_grad)

    def get_feat(self, state):
        """
        Get 1-D feature vector from RSSM state (pooled for patch dynamics).

        Used by both policy (navthinker_policy.py) and trainer (navthinker_trainer.py)
        for reward/trajectory decoders and RL policy.
        """
        return self.dynamics.get_feat(state)

    def get_patch_feat(self, state):
        """
        Get patch-level features (for DINO-WM dynamics).

        Returns (B, [T,] N, D) for depth decoder and feature loss.
        Falls back to get_feat().unsqueeze(-2) for non-patch dynamics.
        """
        if hasattr(self.dynamics, 'get_patch_feat'):
            return self.dynamics.get_patch_feat(state)
        return self.dynamics.get_feat(state).unsqueeze(-2)

    def get_pred_patch_feat(self, state):
        """
        Get predictor-predicted patch features (DINO-WM, for visualization).

        Falls back to get_patch_feat for GRU-based dynamics.
        """
        if hasattr(self.dynamics, 'get_pred_patch_feat'):
            return self.dynamics.get_pred_patch_feat(state)
        return self.get_patch_feat(state)

    # ------------------------------------------------------------------
    # Imagination rollout
    # ------------------------------------------------------------------
    @torch.no_grad()
    def imagine(self, start_state, actor_fn, horizon, human_state_goal=None):
        """Imagine future trajectories in latent space using the prior.

        This is the core of DreamerV3-style imagination: starting from a real
        posterior state, roll forward *H* steps using only the prior (no
        observations).  At each step the ``actor_fn`` selects an action given
        the current RSSM feature.

        Args:
            start_state: RSSM state dict ``{stoch, deter, logit, ...}``
                with shape ``(B, ...)``.
            actor_fn: callable ``(feat) -> action_onehot (B, num_actions)``
            horizon: number of imagination steps *H*.
            human_state_goal: optional ``(B, N, goal_dim)`` held constant
                throughout the rollout (used by the trajectory decoder).

        Returns:
            dict with:
                ``feats``   – ``(B, H, feat_dim)``
                ``rewards`` – ``(B, H, 1)``
                ``traj``    – ``(B, H, N, T, 2)`` predicted human trajectories
                ``states``  – list of H RSSM state dicts
        """
        feats, rewards, trajs, states = [], [], [], []
        state = {k: v.detach() for k, v in start_state.items()}

        # Reset caches for dynamics
        if hasattr(self.dynamics, 'reset_imagination'):
            self.dynamics.reset_imagination()
        if hasattr(self.dynamics, 'reset_kv_cache'):
            B = state.get("stoch", state.get("deter")).shape[0]
            self.dynamics.reset_kv_cache(B)

        for _ in range(horizon):
            feat = self.dynamics.get_feat(state)
            action = actor_fn(feat)

            state = self.dynamics.img_step(state, action)
            feat = self.dynamics.get_feat(state)

            reward_dist = self.heads["reward"](feat)
            traj_dist = self.heads["human_traj"](
                feat, human_state_goal=human_state_goal,
            )

            feats.append(feat)
            rewards.append(reward_dist.mean())
            trajs.append(traj_dist.mean())
            states.append(state)

        return {
            "feats": torch.stack(feats, dim=1),       # (B, H, feat)
            "rewards": torch.stack(rewards, dim=1),    # (B, H, 1)
            "traj": torch.stack(trajs, dim=1),         # (B, H, N, T, 2)
            "states": states,
        }
