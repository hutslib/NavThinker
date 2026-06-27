# Depth Anything V2 Encoder Adapter for World Model
#
# Uses a frozen DINOv2 ViT backbone (from Depth Anything V2) to encode
# RGB observations into rich visual features for the World Model.
#
# Architecture:
#   RGB (B, H, W, 3) → ViT (frozen) → patch features (B, N_patches, D_vit)
#                                     → global pool → 1D feature for RSSM
#                                     → multi-scale reshape for depth decoder
#
# The ViT features serve dual purposes:
#   1. Compressed 1D embedding for RSSM dynamics
#   2. Feature-level supervision target for the feature decoder

import logging
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from habitat_baselines.rl.world_model.networks import (
    VISUAL_KEYS_DEFAULT,
    FUSE_KEYS_1D_DEFAULT,
    HUMAN_STATE_GOAL_KEYS_DEFAULT,
)

logger = logging.getLogger(__name__)


class DepthAnythingEncoderAdapter(nn.Module):
    """
    Frozen Depth Anything V2 (DINOv2 ViT) encoder for World Model.

    Encodes RGB images through a frozen ViT backbone, producing:
      - 1D feature vector for RSSM (via projection of [CLS] + global pool)
      - Patch-level features for feature decoder supervision
      - Multi-scale 2D feature maps for depth decoder skip connections

    Also encodes 1D sensors and human state/goal (same as NavThinkerEncoderAdapter).
    """

    def __init__(
        self,
        observation_space,
        vit_model_name="vits",
        vit_pretrained_path=None,
        hidden_size=256,
        projection_dim=512,
        vit_input_size=518,
        visual_keys=None,
        fuse_keys_1d=None,
        human_state_goal_keys=None,
    ):
        """
        Args:
            observation_space: Habitat observation space
            vit_model_name: ViT variant ('vits', 'vitb', 'vitl')
            vit_pretrained_path: path to DA-V2 pretrained checkpoint
            hidden_size: hidden size for 1D sensor encoders
            projection_dim: output dim of visual projection
            vit_input_size: input resolution for ViT (518 for DA-V2)
        """
        super().__init__()
        _allow_visual = visual_keys if visual_keys is not None else VISUAL_KEYS_DEFAULT
        _allow_1d = fuse_keys_1d if fuse_keys_1d is not None else FUSE_KEYS_1D_DEFAULT
        _allow_hs = human_state_goal_keys if human_state_goal_keys is not None else HUMAN_STATE_GOAL_KEYS_DEFAULT

        self._vit_input_size = vit_input_size
        self._projection_dim = projection_dim

        # ==================== Part 1: Visual Keys ====================
        self._visual_keys = [
            k for k, v in observation_space.spaces.items()
            if len(v.shape) == 3 and any(allow in k for allow in _allow_visual)
        ]

        self.key_needs_rescaling = {k: None for k in self._visual_keys}
        for k, v in observation_space.spaces.items():
            if k in self._visual_keys and v.dtype == np.uint8:
                self.key_needs_rescaling[k] = 1.0 / v.high.max()

        # ==================== Part 2: ViT Backbone ====================
        vit_configs = {
            "vits": {"embed_dim": 384, "num_heads": 6, "depth": 12, "patch_size": 14},
            "vitb": {"embed_dim": 768, "num_heads": 12, "depth": 12, "patch_size": 14},
            "vitl": {"embed_dim": 1024, "num_heads": 16, "depth": 24, "patch_size": 14},
        }
        assert vit_model_name in vit_configs, f"Unknown ViT: {vit_model_name}"
        vit_cfg = vit_configs[vit_model_name]
        self._vit_embed_dim = vit_cfg["embed_dim"]
        self._vit_patch_size = vit_cfg["patch_size"]

        self._vit = self._build_dinov2_vit(vit_cfg)
        self._vit_num_patches = (vit_input_size // self._vit_patch_size) ** 2

        if vit_pretrained_path:
            self._load_vit_weights(vit_pretrained_path)

        # Freeze ViT
        for p in self._vit.parameters():
            p.requires_grad_(False)
        self._vit.eval()

        # Projection: ViT features → compact embedding for RSSM
        visual_outdim = 0
        if len(self._visual_keys) > 0:
            self._visual_projection = nn.Sequential(
                nn.Linear(self._vit_embed_dim, projection_dim, bias=False),
                nn.LayerNorm(projection_dim, eps=1e-3),
                nn.SiLU(),
            )
            visual_outdim = projection_dim

        # ==================== Part 3: 1D Sensors ====================
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
                nn.LayerNorm(hidden_size, eps=1e-3),
                nn.SiLU(),
            )
            sensor_outdim = hidden_size
        else:
            self._fuse_encoder = None

        # ==================== Part 4: Human state+goal ====================
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
                nn.LayerNorm(hidden_size // 2, eps=1e-3),
                nn.SiLU(),
            )
            hs_outdim = hidden_size // 2
        else:
            self._human_state_goal_encoder = None

        self.outdim = visual_outdim + sensor_outdim + hs_outdim

        # Cached features for decoder
        self._cached_vit_patch_features = None

        logger.info("DepthAnythingEncoderAdapter initialized:")
        logger.info(f"  - ViT: {vit_model_name} (embed_dim={self._vit_embed_dim}, frozen)")
        logger.info(f"  - Visual projection: {self._vit_embed_dim} → {projection_dim}")
        logger.info(f"  - 1D Sensors: {sensor_outdim}")
        logger.info(f"  - Human state+goal: {hs_outdim}")
        logger.info(f"  - Total outdim: {self.outdim}")

    @property
    def is_blind(self):
        return len(self._visual_keys) == 0

    @property
    def vit_embed_dim(self):
        return self._vit_embed_dim

    @property
    def vit_num_patches(self):
        return self._vit_num_patches

    def _build_dinov2_vit(self, cfg):
        """Build a DINOv2-style ViT (same arch as Depth Anything V2 encoder)."""
        try:
            from functools import partial
            vit = torch.hub.load(
                'facebookresearch/dinov2', f'dinov2_{self._get_dinov2_name(cfg)}',
                pretrained=False,
            )
            logger.info("DINOv2 ViT-%s loaded via torch.hub (native architecture)",
                        self._get_dinov2_name(cfg))
        except Exception:
            logger.warning("Failed to load DINOv2 from torch.hub, building manually")
            vit = self._build_vit_manual(cfg)
        return vit

    @staticmethod
    def _get_dinov2_name(cfg):
        dim_to_name = {384: "vits14", 768: "vitb14", 1024: "vitl14"}
        return dim_to_name.get(cfg["embed_dim"], "vits14")

    def _build_vit_manual(self, cfg):
        """Fallback manual ViT construction if torch.hub fails.

        torchvision VisionTransformer.forward() returns (B, num_classes),
        not the patch sequence.  We add forward_features() that returns
        (B, 1+N_patches, hidden_dim) so _extract_vit_features can get tokens.
        """
        import torch as _torch
        from torchvision.models.vision_transformer import VisionTransformer
        vit = VisionTransformer(
            image_size=self._vit_input_size,
            patch_size=cfg["patch_size"],
            num_layers=cfg["depth"],
            num_heads=cfg["num_heads"],
            hidden_dim=cfg["embed_dim"],
            mlp_dim=cfg["embed_dim"] * 4,
        )

        # Expose encoder sequence output: (B, 1+N, D)
        def forward_features(x):
            x = vit._process_input(x)
            batch_cls = vit.class_token.expand(x.shape[0], -1, -1)
            x = _torch.cat([batch_cls, x], dim=1)
            x = vit.encoder(x)
            return x  # (B, 1+N, hidden_dim)

        vit.forward_features = forward_features
        return vit

    def _load_vit_weights(self, path):
        """Load pretrained weights (DA-V2 or DINOv2 checkpoint)."""
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"DA-V2 pretrained weights not found:\n  {path}\n"
                "Please download Depth-Anything-V2-Small and save as:\n"
                "  pretrained_model/depth_anything_v2_vits.pth\n"
                "See docs/INSTALL_TODO.md for the download link."
            )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            # DA-V2 checkpoints store encoder weights under various keys
            state = None
            for prefix in ["pretrained.", "encoder.", "model.", ""]:
                candidate = {
                    k[len(prefix):]: v for k, v in ckpt.items()
                    if k.startswith(prefix)
                } if prefix else ckpt
                if len(candidate) > 0:
                    state = candidate
                    break
            if state is None:
                state = ckpt
        else:
            state = ckpt

        missing, unexpected = self._vit.load_state_dict(state, strict=False)
        loaded = len(state) - len(missing)
        logger.info(
            f"[DA-V2 Encoder] Loaded ViT weights from {path}: "
            f"{loaded}/{len(state)} layers, "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )

    def _preprocess_rgb(self, observations):
        """Extract and preprocess RGB for ViT input."""
        imgs = []
        for k in self._visual_keys:
            obs_k = observations[k]
            if obs_k.dim() == 4:
                obs_k = obs_k.permute(0, 3, 1, 2).float()
            if self.key_needs_rescaling[k] is not None:
                obs_k = obs_k * self.key_needs_rescaling[k]
            elif obs_k.max() > 1.0:
                obs_k = obs_k / 255.0
            imgs.append(obs_k)

        x = torch.cat(imgs, dim=1)  # (B, C, H, W)

        # ViT expects 3-channel input; if depth (1ch), repeat to 3ch
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            x = x[:, :3]

        # Resize to ViT input size
        if x.shape[-2] != self._vit_input_size or x.shape[-1] != self._vit_input_size:
            x = F.interpolate(x, size=(self._vit_input_size, self._vit_input_size),
                              mode="bilinear", align_corners=False)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape(1, 3, 1, 1)
        x = (x - mean) / std
        return x

    @torch.no_grad()
    def _extract_vit_features(self, x):
        """
        Extract patch features from frozen ViT.

        Returns:
            patch_features: (B, N_patches, D_vit)
            cls_token: (B, D_vit)
        """
        self._vit.eval()

        if hasattr(self._vit, 'forward_features'):
            # DINOv2 API
            features = self._vit.forward_features(x)
            if isinstance(features, dict):
                patch_features = features.get("x_norm_patchtokens", features.get("x_patchtokens"))
                cls_token = features.get("x_norm_clstoken", features.get("x_clstoken"))
                if patch_features is None:
                    all_tokens = features.get("x_norm", features.get("x"))
                    cls_token = all_tokens[:, 0]
                    patch_features = all_tokens[:, 1:]
            else:
                cls_token = features[:, 0]
                patch_features = features[:, 1:]
        elif hasattr(self._vit, 'get_intermediate_layers'):
            out = self._vit.get_intermediate_layers(x, n=1)[0]
            cls_token = out[:, 0]
            patch_features = out[:, 1:]
        else:
            out = self._vit(x)
            if out.dim() == 3:
                cls_token = out[:, 0]
                patch_features = out[:, 1:]
            else:
                cls_token = out
                patch_features = None

        return patch_features, cls_token

    def forward(self, observations):
        """
        Forward pass. Returns 1D feature vector (B, outdim) for RSSM.

        Side effects:
            Caches self._cached_vit_patch_features for feature decoder loss.
        """
        return self._forward_global(observations)

    def _forward_global(self, observations):
        """Original forward: returns (B, outdim) global vector."""
        outputs = []

        if not self.is_blind:
            x = self._preprocess_rgb(observations)
            patch_features, cls_token = self._extract_vit_features(x)

            self._cached_vit_patch_features = patch_features

            if cls_token is not None:
                global_feat = cls_token
            else:
                global_feat = patch_features.mean(dim=1)

            visual_feat = self._visual_projection(global_feat)
            outputs.append(visual_feat)

        if self._fuse_encoder is not None:
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys_1d], dim=-1
            )
            outputs.append(self._fuse_encoder(fuse_states.float()))

        if (self._human_state_goal_encoder is not None
                and self._human_state_goal_key is not None
                and self._human_state_goal_key in observations):
            hs_obs = observations[self._human_state_goal_key]
            hs_flat = hs_obs.reshape(hs_obs.shape[0], -1)
            outputs.append(self._human_state_goal_encoder(hs_flat.float()))

        if len(outputs) == 0:
            return None
        return torch.cat(outputs, dim=-1)

