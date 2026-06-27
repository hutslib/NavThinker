# Copyright (c) 2024 ForeSightNav-WM
# World Model Neural Network Architecture for Social Navigation
# Includes RSSM, Encoders, and Decoders for depth and human trajectory prediction

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

from habitat_baselines.rl.world_model import tools


class RSSM(nn.Module):
    """
    Recurrent State-Space Model (RSSM) for世界模型
    维护deterministic memory state (h_t) 和 stochastic latent state (z_t)
    """
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=32,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=4,
        embed_size=512,
        device="cuda",
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        act = getattr(torch.nn, act)
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed_size
        self._device = device

        # Input layers for imagination step
        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(act())
        self._img_in_layers = nn.Sequential(*inp_layers)
        self._img_in_layers.apply(tools.weight_init)

        # GRU cell for recurrent state
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)

        # Output layers for prior prediction
        img_out_layers = []
        img_out_layers.append(nn.Linear(self._deter, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(act())
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        # Output layers for posterior inference
        obs_out_layers = []
        inp_dim = self._deter + self._embed
        obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        # Statistical layers for stochastic state
        if self._discrete:
            self._imgs_stat_layer = nn.Linear(
                self._hidden, self._stoch * self._discrete
            )
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))
        else:
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        # Learnable initial state
        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

    def initial(self, batch_size):
        """Initialize RSSM state"""
        deter = torch.zeros(batch_size, self._deter).to(self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )

        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        """Observe sequence and compute posterior and prior"""
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)

        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )

        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine_with_action(self, action, state):
        """Imagine forward using actions"""
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        """Get feature vector from state"""
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_deter_feat(self, state):
        """Get deterministic feature"""
        return state["deter"]

    def get_dist(self, state, dtype=None):
        """Get distribution from state statistics"""
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        """Single step posterior inference"""
        # Ensure is_first is (batch,) for this time step so broadcast with prev_action (batch, num_actions) works
        is_first = is_first.reshape(-1)[: prev_action.shape[0]].to(prev_action.dtype)
        batch_size = is_first.shape[0]
        if prev_state == None or torch.sum(is_first) == batch_size:
            prev_state = self.initial(batch_size)
            prev_action = torch.zeros((batch_size, self._num_actions)).to(self._device)
        elif torch.sum(is_first) > 0:
            is_first = is_first.reshape(-1, 1)  # (batch, 1) for broadcast with (batch, num_actions)
            prev_action = prev_action * (1.0 - is_first)
            init_state = self.initial(batch_size)
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], -1)
        x = self._obs_out_layers(x)
        stats = self._suff_stats_layer("obs", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        """Single step imagination/prior prediction"""
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            prev_stoch = prev_stoch.reshape(shape)

        x = torch.cat([prev_stoch, prev_action], -1)
        x = self._img_in_layers(x)

        for _ in range(self._rec_depth):
            deter = prev_state["deter"]
            x, deter = self._cell(x, [deter])
            deter = deter[0]

        x = self._img_out_layers(x)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter):
        """Get stochastic state from deterministic state"""
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        """Compute sufficient statistics for distribution"""
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        """Compute KL divergence loss"""
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )

        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


# 直接指定 WM 编码的观测 key 子串（与 observation_space 中 key 做子串匹配）
VISUAL_KEYS_DEFAULT = ("articulated_agent_jaw_depth",)  # 3D (H,W,C) 颌部深度图
FUSE_KEYS_1D_DEFAULT = ("pointgoal_with_gps_compass", "human_num_sensor")  # 1D
HUMAN_STATE_GOAL_KEYS_DEFAULT = ("human_state_goal",)  # 2D (N,8)，融合状态+目标


class SocialNavEncoder(nn.Module):
    """
    Encoder for Social Navigation observations
    参考Habitat的ResNetEncoder，编码visual和navigation sensors
    所有参与编码的观测均通过 allowlist 直接指定（visual / 1D / state_goal）。
    """
    def __init__(
        self,
        observation_space,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
        hidden_size=256,
        visual_keys=None,
        fuse_keys_1d=None,
        human_state_goal_keys=None,
    ):
        super(SocialNavEncoder, self).__init__()
        act_fn = getattr(torch.nn, act)
        _allow_visual = visual_keys if visual_keys is not None else VISUAL_KEYS_DEFAULT
        _allow_1d = fuse_keys_1d if fuse_keys_1d is not None else FUSE_KEYS_1D_DEFAULT
        _allow_hs = human_state_goal_keys if human_state_goal_keys is not None else HUMAN_STATE_GOAL_KEYS_DEFAULT

        # 视觉：直接指定 3D key 子串，且 shape 为 (H, W, C)
        self.visual_keys = [
            k for k, v in observation_space.spaces.items()
            if len(v.shape) == 3 and any(allow in k for allow in _allow_visual)
        ]

        # Check which keys need rescaling (uint8 -> float)
        self.key_needs_rescaling = {k: None for k in self.visual_keys}
        for k, v in observation_space.spaces.items():
            if k in self.visual_keys and v.dtype == np.uint8:
                self.key_needs_rescaling[k] = 1.0 / v.high.max()

        # Count total visual channels
        self._n_input_channels = sum(
            observation_space.spaces[k].shape[2]
            for k in self.visual_keys
        ) if self.visual_keys else 0

        self.outdim = 0

        # Visual encoder (CNN for depth/rgb)
        if not self.is_blind:
            h, w = observation_space.spaces[self.visual_keys[0]].shape[:2]
            # IMPORTANT: forward() does avg_pool2d(x, 2) before encoder
            # So the actual input to ConvEncoder is half the size
            input_shape = (h // 2, w // 2, self._n_input_channels)

            self._visual_encoder = ConvEncoder(
                input_shape, depth, act, norm, kernel_size, minres
            )
            self.outdim += self._visual_encoder.outdim

        # 1D：直接指定 key 子串
        self._fuse_keys_1d = [
            k for k in observation_space.spaces.keys()
            if len(observation_space.spaces[k].shape) == 1
            and any(allow in k for allow in _allow_1d)
        ]

        if len(self._fuse_keys_1d) > 0:
            fuse_dim = sum(
                observation_space.spaces[k].shape[0]
                for k in self._fuse_keys_1d
            )
            self._fuse_encoder = nn.Sequential(
                nn.Linear(fuse_dim, hidden_size, bias=False),
                nn.LayerNorm(hidden_size, eps=1e-03) if norm else nn.Identity(),
                act_fn(),
            )
            self._fuse_encoder.apply(tools.weight_init)
            self.outdim += hidden_size

        # Human state+goal：直接指定 key 子串，2D (N, 8)
        self._human_state_goal_key = next(
            (k for k in observation_space.spaces.keys()
             if len(observation_space.spaces[k].shape) == 2
             and any(allow in k for allow in _allow_hs)),
            None,
        )

        if self._human_state_goal_key is not None:
            hs_shape = observation_space.spaces[self._human_state_goal_key].shape
            hs_flatten_dim = hs_shape[0] * hs_shape[1]
            self._human_state_goal_encoder = nn.Sequential(
                nn.Linear(hs_flatten_dim, hidden_size // 2, bias=False),
                nn.LayerNorm(hidden_size // 2, eps=1e-03) if norm else nn.Identity(),
                act_fn(),
            )
            self._human_state_goal_encoder.apply(tools.weight_init)
            self.outdim += hidden_size // 2
        else:
            self._human_state_goal_encoder = None

    @property
    def is_blind(self):
        return self._n_input_channels == 0

    def forward(self, observations):
        """
        Forward pass - compatible with Habitat observation dict

        Args:
            observations: dict with keys like depth, gps, compass,
                         human_state_goal, etc.
        """
        outputs = []

        # Process visual inputs
        if not self.is_blind:
            cnn_input = []
            for k in self.visual_keys:
                obs_k = observations[k]
                # permute to [BATCH x CHANNEL x HEIGHT x WIDTH]
                obs_k = obs_k.permute(0, 3, 1, 2)
                if self.key_needs_rescaling[k] is not None:
                    obs_k = obs_k.float() * self.key_needs_rescaling[k]
                cnn_input.append(obs_k)

            x = torch.cat(cnn_input, dim=1)
            x = F.avg_pool2d(x, 2)
            visual_feats = self._visual_encoder.forward_features(x)
            outputs.append(visual_feats)

        # Process 1D sensors (GPS, Compass, etc.)
        if len(self._fuse_keys_1d) > 0:
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys_1d], dim=-1
            )
            fuse_feats = self._fuse_encoder(fuse_states.float())
            outputs.append(fuse_feats)

        # Process human state+goal fusion sensor (agent_0_human_state_goal)
        if self._human_state_goal_encoder is not None and self._human_state_goal_key is not None and self._human_state_goal_key in observations:
            hs_obs = observations[self._human_state_goal_key]
            hs_flat = hs_obs.reshape(hs_obs.shape[0], -1)
            hs_feats = self._human_state_goal_encoder(hs_flat.float())
            outputs.append(hs_feats)

        if len(outputs) == 0:
            return None

        return torch.cat(outputs, dim=-1)


class DepthDecoder(nn.Module):
    """
    Depth Reconstruction Decoder using ConvTranspose upsampling.

    Used with GRU-based RSSM dynamics (1D feature input).
    """
    def __init__(
        self,
        feat_size,
        depth_shape=(256, 256, 1),
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
        outscale=1.0,
    ):
        super(DepthDecoder, self).__init__()

        self._depth_decoder = ConvDecoder(
            feat_size,
            shape=(depth_shape[2], depth_shape[0], depth_shape[1]),
            depth=depth,
            act=act,
            norm=norm,
            kernel_size=kernel_size,
            minres=minres,
            outscale=outscale,
            cnn_sigmoid=True,
        )

    def forward(self, features):
        """
        Args:
            features: RSSM features (batch, [time,] feat_dim)
        """
        return self._depth_decoder(features)


class ViTFeatureDecoder(nn.Module):
    """
    Predicts ViT patch-level features from RSSM latent state.

    Used for feature-level supervision: the frozen DA-V2 ViT produces
    target patch features, and this decoder tries to reconstruct them
    from the world model's latent representation.

    Architecture:
        feat (B, feat_dim) → MLP → (B, N_patches, vit_embed_dim)
    """
    def __init__(
        self,
        feat_size,
        vit_embed_dim=384,
        num_patches=1369,
        hidden_size=512,
        num_layers=2,
        act="SiLU",
        norm=True,
    ):
        super().__init__()
        self._vit_embed_dim = vit_embed_dim
        self._num_patches = num_patches
        act_fn = getattr(torch.nn, act)

        layers = []
        in_dim = feat_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size, bias=False))
            if norm:
                layers.append(nn.LayerNorm(hidden_size, eps=1e-3))
            layers.append(act_fn())
            in_dim = hidden_size

        self._backbone = nn.Sequential(*layers)
        self._backbone.apply(tools.weight_init)

        self._out = nn.Linear(hidden_size, num_patches * vit_embed_dim)
        self._out.apply(tools.uniform_weight_init(1.0))

    def forward(self, features):
        """
        Args:
            features: (B, [T,] feat_dim)
        Returns:
            Predicted patch features: (B, [T,] N_patches, vit_embed_dim)
        """
        leading_shape = features.shape[:-1]
        x = self._backbone(features)
        x = self._out(x)
        x = x.reshape(*leading_shape, self._num_patches, self._vit_embed_dim)
        return x


class HumanTrajectoryDecoder(nn.Module):
    """
    Per-human trajectory prediction decoder with residual output.

    Architecture:
      1. Per-human goal encoder: (N, 8) → (N, H)  — two-layer MLP, shared weights
      2. Per-human MLP: concat(feat_broadcast, goal_h) → (N, H) → (N, T*2)
      3. Residual: output = delta + anchor_pos

    Each human's trajectory is decoded independently conditioned on the shared
    world-model feature and its own goal/state vector. This preserves the
    identity mapping between goal info and trajectory output.
    """
    def __init__(
        self,
        feat_size,
        num_humans=10,
        pred_horizon=12,
        traj_dim=2,
        hidden_layers=3,
        hidden_units=256,
        act="SiLU",
        norm=True,
        outscale=1.0,
        use_goal_conditioning=True,
        state_goal_dim=8,
        residual=True,
    ):
        super(HumanTrajectoryDecoder, self).__init__()
        self.num_humans = num_humans
        self.pred_horizon = pred_horizon
        self.traj_dim = traj_dim
        self.state_goal_dim = state_goal_dim
        self.use_goal_conditioning = use_goal_conditioning
        self.residual = residual

        act_fn = getattr(torch.nn, act)

        # Per-human goal encoder (shared across all humans)
        if use_goal_conditioning:
            self.goal_per_human = nn.Sequential(
                nn.Linear(state_goal_dim, hidden_units, bias=False),
                nn.LayerNorm(hidden_units, eps=1e-03) if norm else nn.Identity(),
                act_fn(),
                nn.Linear(hidden_units, hidden_units, bias=False),
                nn.LayerNorm(hidden_units, eps=1e-03) if norm else nn.Identity(),
                act_fn(),
            )
            self.goal_per_human.apply(tools.weight_init)
            mlp_inp_dim = feat_size + hidden_units
        else:
            self.goal_per_human = None
            mlp_inp_dim = feat_size

        # Per-human trajectory MLP (shared across all humans)
        layers = []
        inp_dim = mlp_inp_dim
        for i in range(hidden_layers):
            layers.append(nn.Linear(inp_dim, hidden_units, bias=False))
            if norm:
                layers.append(nn.LayerNorm(hidden_units, eps=1e-03))
            layers.append(act_fn())
            inp_dim = hidden_units

        self._mlp = nn.Sequential(*layers)
        self._mlp.apply(tools.weight_init)

        self._out_layer = nn.Linear(hidden_units, pred_horizon * traj_dim)
        self._out_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, human_state_goal=None):
        """
        Args:
            features: (batch, [time,] feat_dim)
            human_state_goal: (batch, [time,] num_humans, state_goal_dim)

        Returns:
            MSEDist over trajectory positions (batch, [time,] N, T, 2).
        """
        # Broadcast feat to per-human: (..., feat_dim) → (..., N, feat_dim)
        feat_expanded = features.unsqueeze(-2).expand(
            *features.shape[:-1], self.num_humans, features.shape[-1]
        )

        if self.use_goal_conditioning and human_state_goal is not None:
            goal_h = self.goal_per_human(human_state_goal)   # (..., N, H)
            x = torch.cat([feat_expanded, goal_h], dim=-1)   # (..., N, feat+H)
        else:
            x = feat_expanded

        x = self._mlp(x)                                     # (..., N, H)
        traj_delta = self._out_layer(x)                       # (..., N, T*2)
        traj_delta = traj_delta.reshape(
            *traj_delta.shape[:-1], self.pred_horizon, self.traj_dim
        )                                                     # (..., N, T, 2)

        if self.residual and human_state_goal is not None:
            anchor = human_state_goal[..., :self.traj_dim]    # (..., N, 2)
            traj_mean = traj_delta + anchor.unsqueeze(-2)     # (..., N, T, 2)
        else:
            traj_mean = traj_delta

        return tools.MSEDist(traj_mean, agg="mean")


class RewardDecoder(nn.Module):
    """
    Reward Prediction Decoder
    预测 reward 值，返回 MSEDist distribution
    """
    def __init__(
        self,
        feat_size,
        hidden_size=256,
        act="SiLU",
        norm=True,
        outscale=1.0,
    ):
        super(RewardDecoder, self).__init__()
        act_fn = getattr(torch.nn, act)

        self._net = nn.Sequential(
            nn.Linear(feat_size, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=1e-03) if norm else nn.Identity(),
            act_fn(),
            nn.Linear(hidden_size, 1),
        )

        # Apply weight initialization
        self._net[0].apply(tools.weight_init)
        self._net[-1].apply(tools.uniform_weight_init(outscale))

    def forward(self, features):
        """
        Args:
            features: World model features (batch, [time,] feat_dim)

        Returns:
            Distribution over reward values
        """
        reward_mean = self._net(features)
        return tools.MSEDist(reward_mean, agg="sum")


class ConvEncoder(nn.Module):
    """Convolutional Encoder for image-like inputs"""
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        stages = int(np.log2(w) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        layers = []

        for i in range(stages):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act())
            in_dim = out_dim
            out_dim *= 2
            h, w = (h + 1) // 2, (w + 1) // 2

        self.outdim = out_dim // 2 * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        """
        Forward pass for world model training
        Expects: (batch, time, h, w, ch)
        """
        # Normalize if needed
        if obs.max() > 1.0:
            obs = obs / 255.0
        obs = obs - 0.5
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])

    def forward_features(self, x):
        """
        Forward pass for already preprocessed features
        Expects: (batch, ch, h, w) - already pooled and in correct format
        Used by SocialNavEncoder
        """
        x = x - 0.5  # Normalize
        x = self.layers(x)
        # Flatten spatial dimensions
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        return x


class ConvDecoder(nn.Module):
    """Convolutional Decoder for image-like outputs"""
    def __init__(
        self,
        feat_size,
        shape=(1, 256, 256),
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
    ):
        super(ConvDecoder, self).__init__()
        act = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid

        # Calculate layer sizes
        input_ch, h, w = shape
        stages = int(np.log2(w) - np.log2(minres))
        self.h_list = []
        self.w_list = []
        for i in range(stages):
            h, w = (h + 1) // 2, (w + 1) // 2
            self.h_list.append(h)
            self.w_list.append(w)
        self.h_list = self.h_list[::-1]
        self.w_list = self.w_list[::-1]
        self.h_list.append(shape[1])
        self.w_list.append(shape[2])

        layer_num = len(self.h_list) - 1
        out_ch = self.h_list[0] * self.w_list[0] * depth * 2 ** (len(self.h_list) - 2)
        self._embed_size = out_ch

        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(tools.uniform_weight_init(outscale))

        in_dim = out_ch // (self.h_list[0] * self.w_list[0])
        out_dim = in_dim // 2

        layers = []
        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth

            if self.h_list[i] * 2 == self.h_list[i + 1]:
                pad_h, outpad_h = 1, 0
            else:
                pad_h, outpad_h = 2, 1

            if self.w_list[i] * 2 == self.w_list[i + 1]:
                pad_w, outpad_w = 1, 0
            else:
                pad_w, outpad_w = 2, 1

            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            in_dim = out_dim
            out_dim //= 2

        [m.apply(tools.weight_init) for m in layers[:-1]]
        layers[-1].apply(tools.uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        """Forward pass"""
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self.h_list[0], self.w_list[0],
             self._embed_size // (self.h_list[0] * self.w_list[0])]
        )
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ch, h, w) -> (batch * time, h, w, ch)
        mean = x.permute(0, 2, 3, 1)
        # (batch * time, h, w, ch) -> (batch, time, h, w, ch)
        mean = mean.reshape(features.shape[:-1] + self._shape[1:] + (self._shape[0],))

        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)

        return mean


class GRUCell(nn.Module):
    """Gated Recurrent Unit Cell"""
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


# ---------------------------------------------------------------------------
# DPT-style Depth Head for patch-level latent decoding
# ---------------------------------------------------------------------------

class DPTDepthHead(nn.Module):
    """
    DPT-style depth decoder that operates on patch-level latent features.

    Takes (B, N, D) patch tokens from DINO-WM dynamics, reshapes them
    into a 2D spatial map, and progressively upsamples to the target depth
    resolution.
    """

    def __init__(
        self,
        patch_feat_dim,
        num_patches_h=16,
        num_patches_w=16,
        depth_shape=(256, 256, 1),
        features=64,
        act="SiLU",
        norm=True,
    ):
        super().__init__()
        self._depth_shape = depth_shape
        self._h = num_patches_h
        self._w = num_patches_w
        self._features = features
        act_fn = getattr(torch.nn, act)

        self._input_proj = nn.Sequential(
            nn.Linear(patch_feat_dim, features * 4, bias=False),
            nn.LayerNorm(features * 4, eps=1e-3),
            act_fn(),
        )
        self._input_proj.apply(tools.weight_init)

        # Progressive upsampling stages
        self._up_stages = nn.ModuleList()
        channels = [features * 4, features * 2, features, features // 2]
        for i in range(len(channels) - 1):
            stage = nn.Sequential(
                nn.ConvTranspose2d(channels[i], channels[i + 1], 4, 2, 1, bias=False),
                ImgChLayerNorm(channels[i + 1]) if norm else nn.Identity(),
                act_fn(),
            )
            stage.apply(tools.weight_init)
            self._up_stages.append(stage)

        final_in_ch = channels[-1]
        target_h, target_w = depth_shape[0], depth_shape[1]
        self._final_up = nn.Sequential(
            nn.Conv2d(final_in_ch, final_in_ch, 3, 1, 1, bias=False),
            ImgChLayerNorm(final_in_ch) if norm else nn.Identity(),
            act_fn(),
            nn.Conv2d(final_in_ch, 1, 1),
        )
        self._final_up.apply(tools.weight_init)
        self._target_h = target_h
        self._target_w = target_w

    def forward(self, patch_features):
        """
        Args:
            patch_features: (B, [T,] N, D) patch-level latent from DINO-WM

        Returns:
            MSEDist over depth: mean shape (B, [T,] H, W, 1)
        """
        leading_shape = patch_features.shape[:-2]
        N, D = patch_features.shape[-2], patch_features.shape[-1]
        flat = patch_features.reshape(-1, N, D)
        BT = flat.shape[0]

        x = self._input_proj(flat)
        x = x.permute(0, 2, 1).reshape(BT, -1, self._h, self._w)

        for stage in self._up_stages:
            x = stage(x)

        if x.shape[2] != self._target_h or x.shape[3] != self._target_w:
            x = F.interpolate(
                x, size=(self._target_h, self._target_w),
                mode="bilinear", align_corners=False,
            )

        x = self._final_up(x)
        mean = x.permute(0, 2, 3, 1)
        mean = F.sigmoid(mean)

        mean = mean.reshape(*leading_shape, self._target_h, self._target_w, 1)
        return tools.MSEDist(mean, agg="mean")



class Conv2dSamePad(torch.nn.Conv2d):
    """Conv2d with 'SAME' padding"""
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ImgChLayerNorm(nn.Module):
    """Layer normalization for image channels"""
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
