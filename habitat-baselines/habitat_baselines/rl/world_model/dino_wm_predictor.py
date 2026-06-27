# DINO-WM style Causal ViT Predictor for World Model dynamics.
#
# Adapted from: https://github.com/gaoyuezhou/dino_wm
# Paper: "DINO-WM: World Models on Pre-trained Visual Features enable
#         Zero-shot Planning" (ICML 2025)
#
# Core architecture directly reused from DINO-WM with minimal changes:
#   - vit.py: ViTPredictor, Transformer, Attention, FeedForward
#   - proprio.py: ProprioceptiveEmbedding (used as action encoder)
#   - visual_world_model.py: encode/predict/separate_emb/rollout logic
#
# Changes from original DINO-WM:
#   - Causal mask uses instance variables instead of globals
#   - DinoWMDynamics wrapper exposes RSSM-compatible interface
#     (observe / img_step / get_feat / get_patch_feat / get_deter_feat)
#   - No proprio encoder (social nav has no proprioception)
#   - Action is discrete one-hot (not continuous)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# From dino_wm/models/vit.py (adapted: causal mask as instance var)
# =====================================================================

def _generate_causal_mask(num_patches_per_frame, num_frames, device="cpu"):
    """Block-causal mask: frame i can attend to frames 0..i."""
    n = num_patches_per_frame
    total = n * num_frames
    mask = torch.zeros(total, total, device=device)
    for i in range(num_frames):
        end = (i + 1) * n
        mask[i * n : end, :end] = 1.0
    return mask.unsqueeze(0).unsqueeze(0)


class _FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self._causal_mask = None

    def set_causal_mask(self, mask):
        self._causal_mask = mask

    def forward(self, x):
        B, T, C = x.size()
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        h = self.heads
        q, k, v = [
            t.reshape(B, T, h, -1).permute(0, 2, 1, 3) for t in qkv
        ]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if self._causal_mask is not None:
            mask = self._causal_mask[:, :, :T, :T].to(dots.device)
            dots = dots.masked_fill(mask == 0, float("-inf"))
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, T, -1)
        return self.to_out(out)


class _Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        _Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        _FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def set_causal_mask(self, mask):
        for attn, _ in self.layers:
            attn.set_causal_mask(mask)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class _ViTPredictor(nn.Module):
    """Causal ViT predictor from DINO-WM (models/vit.py)."""

    def __init__(
        self,
        num_patches_per_frame,
        num_frames,
        dim,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.0,
    ):
        super().__init__()
        self._num_patches = num_patches_per_frame
        self._num_frames = num_frames
        total_tokens = num_frames * num_patches_per_frame
        self.pos_embedding = nn.Parameter(torch.randn(1, total_tokens, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = _Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        mask = _generate_causal_mask(num_patches_per_frame, num_frames)
        self.register_buffer("_causal_mask", mask)
        self.transformer.set_causal_mask(self._causal_mask)

    def forward(self, x):
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        return x


# =====================================================================
# From dino_wm/models/proprio.py (action encoder)
# =====================================================================

class _ActionEncoder(nn.Module):
    """Maps per-step action to embedding dim via Conv1d (from DINO-WM)."""

    def __init__(self, action_dim, emb_dim):
        super().__init__()
        self.patch_embed = nn.Conv1d(action_dim, emb_dim, kernel_size=1, stride=1)

    def forward(self, x):
        # x: (B, T, action_dim)
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return x  # (B, T, emb_dim)


# =====================================================================
# DinoWMDynamics: RSSM-compatible wrapper
# =====================================================================

class DinoWMDynamics(nn.Module):
    """DINO-WM Causal ViT predictor with RSSM-compatible interface.

    Operates directly in the ViT feature space (e.g. 384-d DA-V2 patch
    tokens) without any stochastic latent variable or information bottleneck.

    State dict layout (RSSM-compatible):
        deter : (B, N_patches, dim)   -- predicted ViT patch tokens
        (no stoch / logit / mean / std)

    The ``concat_dim=0`` strategy from DINO-WM is used: action is
    injected as an extra token appended to each frame's patch sequence.
    """

    def __init__(
        self,
        dim=384,
        num_patches=1369,
        num_hist=3,
        num_pred=1,
        action_dim=4,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.0,
        device="cuda",
    ):
        super().__init__()
        self._dim = dim
        self._num_patches = num_patches
        self._num_hist = num_hist
        self._num_pred = num_pred
        self._action_dim = action_dim
        self._device = device
        self._num_actions = action_dim

        # +1 for the action token appended to each frame
        tokens_per_frame = num_patches + 1
        self._tokens_per_frame = tokens_per_frame

        self.action_encoder = _ActionEncoder(action_dim, dim)

        self.predictor = _ViTPredictor(
            num_patches_per_frame=tokens_per_frame,
            num_frames=num_hist + num_pred,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

        self.emb_criterion = nn.MSELoss()

        # History buffer for imagination rollout
        self._hist_z = None

    # ----------------------------------------------------------------
    # DINO-WM core logic (from visual_world_model.py)
    # ----------------------------------------------------------------

    def _encode_and_concat(self, embed, action):
        """Concatenate visual patches + action token (concat_dim=0).

        Args:
            embed: (B, T, N, D) ViT patch tokens
            action: (B, T, action_dim) one-hot actions

        Returns:
            z: (B, T, N+1, D) with action as extra token
        """
        act_emb = self.action_encoder(action)  # (B, T, D)
        z = torch.cat([embed, act_emb.unsqueeze(2)], dim=2)
        return z

    def _predict(self, z):
        """Run ViT predictor on concatenated tokens.

        Args:
            z: (B, T, tokens_per_frame, D)

        Returns:
            z_pred: (B, T, tokens_per_frame, D)
        """
        B, T, P, D = z.shape
        z = z.reshape(B, T * P, D)
        z = self.predictor(z)
        z = z.reshape(B, T, P, D)
        return z

    def _separate_visual_action(self, z):
        """Split visual patches and action token (concat_dim=0)."""
        visual = z[:, :, :-1, :]  # (B, T, N, D)
        action = z[:, :, -1, :]   # (B, T, D)
        return visual, action

    # ----------------------------------------------------------------
    # RSSM-compatible interface
    # ----------------------------------------------------------------

    def initial(self, batch_size):
        deter = torch.zeros(
            batch_size, self._num_patches, self._dim, device=self._device
        )
        return {"deter": deter}

    def observe(self, embed, action, is_first, state=None):
        """Process a full sequence of observations.

        Follows the original DINO-WM training protocol:
          z_src = z[:, :num_hist]   (first num_hist frames)
          z_tgt = z[:, num_pred:]   (shifted by num_pred, also num_hist frames)
          z_pred = predict(z_src)
          z_loss = MSE(z_pred visual, z_tgt visual)

        Each frame in z_pred is supervised by its corresponding shifted GT
        frame thanks to the block-causal mask:
          pred frame 0 → target frame 1
          pred frame 1 → target frame 2
          pred frame 2 → target frame 3  (when num_hist=3, num_pred=1)

        Args:
            embed: (B, T, N, D) ViT patch tokens from encoder
            action: (B, T, action_dim) one-hot actions
            is_first: (B, T) episode boundary flags (unused for now)
            state: previous state (kept for API compatibility, ignored)

        Returns:
            post, prior: both are the same dict (no stochastic variable)
        """
        if embed.dim() == 3:
            B, T, flat_dim = embed.shape
            embed = embed.reshape(B, T, self._num_patches, self._dim)

        B, T, N, D = embed.shape

        z = self._encode_and_concat(embed, action)  # (B, T, N+1, D)

        out_state = {
            "deter": embed,      # (B, T, N, D) -- GT features (for decoders)
            "z_full": z,         # (B, T, N+1, D) -- for compute_z_loss
        }
        return out_state, out_state

    def compute_pred_deter(self, state):
        """Return predicted patch features (reuses cache from compute_z_loss).

        If ``compute_z_loss`` has already been called on this ``state``,
        the cached ``state["pred_deter"]`` is returned directly — no extra
        predictor forward pass.  Otherwise falls back to computing from
        scratch (e.g. during visualization-only paths).
        """
        cached = state.get("pred_deter")
        if cached is not None:
            return cached

        z_full = state.get("z_full")
        embed = state["deter"]
        if z_full is None:
            return embed

        B, T, N, D = embed.shape
        num_hist = self._num_hist
        pred_deter = embed.clone()
        if T > num_hist:
            for t_start in range(0, T - num_hist):
                z_src = z_full[:, t_start : t_start + num_hist]
                z_pred = self._predict(z_src)
                pred_visual, _ = self._separate_visual_action(z_pred)
                target_t = t_start + num_hist
                if target_t < T:
                    pred_deter[:, target_t] = pred_visual[:, -1]
        return pred_deter

    def img_step(self, prev_state, prev_action, sample=True):
        """Single imagination step (autoregressive).

        Uses the history buffer to maintain context across steps.
        """
        B = prev_action.shape[0]
        prev_deter = prev_state["deter"]  # (B, N, D) or (B, T, N, D)

        if prev_deter.dim() == 3:
            prev_deter = prev_deter.unsqueeze(1)  # (B, 1, N, D)

        # Build action token
        act_emb = self.action_encoder(
            prev_action.unsqueeze(1)
        )  # (B, 1, D)
        z_new = torch.cat(
            [prev_deter[:, -1:], act_emb.unsqueeze(2)], dim=2
        )  # (B, 1, N+1, D)

        # Append to history and predict
        if self._hist_z is None:
            self._hist_z = z_new
        else:
            self._hist_z = torch.cat([self._hist_z, z_new], dim=1)

        # Keep only last num_hist frames
        if self._hist_z.shape[1] > self._num_hist:
            self._hist_z = self._hist_z[:, -self._num_hist:]

        z_pred = self._predict(self._hist_z)
        pred_visual, _ = self._separate_visual_action(z_pred)

        # Take the last predicted frame as the new state
        new_deter = pred_visual[:, -1]  # (B, N, D)

        return {"deter": new_deter}

    def observe_post_action(self, embed, action, is_first):
        """Update ``_hist_z`` with ``(obs_t, action_t)`` **after** the policy
        has chosen ``action_t``.  This keeps every frame in the history buffer
        aligned with the WM training convention where the action token
        represents the action executed *at* that observation, not the action
        that *led to* it.

        Call this once per environment step, after ``act()`` returns.

        Args:
            embed: (B, 1, N, D) or (B, N, D) ViT patch tokens for obs_t
            action: (B, 1, action_dim) or (B, action_dim) one-hot of action_t
            is_first: (B, 1) or (B,) episode boundary flags
        """
        if embed.dim() == 3:
            embed = embed.unsqueeze(1)
        if action.dim() == 2:
            action = action.unsqueeze(1)

        B = embed.shape[0]
        z_new = self._encode_and_concat(embed, action)  # (B, 1, N+1, D)

        is_first_flat = is_first.view(B).bool()

        if is_first_flat.any():
            if self._hist_z is not None:
                mask = is_first_flat.view(B, 1, 1, 1)
                first_frame = z_new.expand_as(self._hist_z[:, :1])
                padded_reset = first_frame.repeat(1, self._hist_z.shape[1], 1, 1)
                self._hist_z = torch.where(mask, padded_reset, self._hist_z)

        if self._hist_z is None:
            self._hist_z = z_new
        else:
            self._hist_z = torch.cat([self._hist_z, z_new], dim=1)

        if self._hist_z.shape[1] > self._num_hist:
            self._hist_z = self._hist_z[:, -self._num_hist:]

    def img_step_lookahead(self, prev_state, prev_action):
        """Side-effect-free imagination step for lookahead planning.

        Identical to ``img_step`` but saves and restores ``_hist_z`` so the
        internal history buffer is not polluted.  This allows the policy to
        query "what would happen if I took action X" without affecting the
        normal WM state progression.
        """
        saved_hist = self._hist_z
        try:
            result = self.img_step(prev_state, prev_action, sample=False)
        finally:
            self._hist_z = saved_hist
        return result

    def on_envs_pause(self, envs_to_pause):
        """Remove paused environments from the history buffer."""
        if self._hist_z is not None and len(envs_to_pause) > 0:
            keep = [i for i in range(self._hist_z.shape[0])
                    if i not in envs_to_pause]
            if len(keep) == 0:
                self._hist_z = None
            else:
                self._hist_z = self._hist_z[keep]

    def reset_imagination(self, batch_size=None):
        """Reset the history buffer for a new imagination rollout."""
        self._hist_z = None

    def get_feat(self, state):
        """1-D pooled feature for reward/trajectory decoders."""
        deter = state["deter"]
        if deter.dim() == 4:
            # (B, T, N, D) -> pool over patches -> (B, T, D)
            return deter.mean(dim=-2)
        # (B, N, D) -> (B, D)
        return deter.mean(dim=-2)

    def get_deter_feat(self, state):
        """Pooled deterministic feature for RL policy."""
        return self.get_feat(state)

    def get_patch_feat(self, state):
        """Full patch-level features for depth decoder (GT)."""
        return state["deter"]

    def get_pred_patch_feat(self, state):
        """Predicted patch-level features from the predictor (for visualization)."""
        return self.compute_pred_deter(state)

    @property
    def feat_size(self):
        return self._dim

    @property
    def patch_feat_size(self):
        return self._dim

    def compute_z_loss(self, state, cached_vit_features, batch_size, seq_len):
        """Compute z_loss and cache predicted patch features (single _predict pass).

        Original DINO-WM protocol: stride-1 windows of num_hist frames,
        each predicted frame supervised by the corresponding shifted GT.

        The predicted visual tokens are also stored in
        ``state["pred_deter"]`` so that ``get_pred_patch_feat`` can reuse
        them without running the predictor a second time.

        Args:
            state: output of observe(), must contain "z_full".
                   **Modified in-place**: ``state["pred_deter"]`` is set.
            cached_vit_features: (B*T, N_vit, D) raw ViT patch features
            batch_size: B
            seq_len: T

        Returns:
            z_loss: scalar MSE loss
        """
        z_full = state.get("z_full")
        if z_full is None:
            return torch.tensor(0.0, device=self._device)

        num_pred = self._num_pred  # typically 1
        num_hist = self._num_hist  # typically 2-3
        window = num_hist + num_pred

        if seq_len < window:
            return torch.tensor(0.0, device=self._device)

        vit_target = cached_vit_features.detach().reshape(
            batch_size, seq_len, *cached_vit_features.shape[1:]
        )

        embed = state["deter"]  # (B, T, N, D)
        pred_deter = embed.clone()

        total_loss = 0.0
        count = 0
        for t_start in range(0, seq_len - window + 1):
            z_src = z_full[:, t_start : t_start + num_hist]  # (B, num_hist, N+1, D)
            z_pred = self._predict(z_src)                     # (B, num_hist, N+1, D)
            pred_visual, _ = self._separate_visual_action(z_pred)  # (B, num_hist, N, D)

            # Cache the last predicted frame for depth_pred_loss / visualization
            target_t = t_start + num_hist
            if target_t < seq_len:
                pred_deter[:, target_t] = pred_visual[:, -1]

            tgt_start = t_start + num_pred
            tgt_end = tgt_start + num_hist
            target = vit_target[:, tgt_start:tgt_end]  # (B, num_hist, N_vit, D)

            if pred_visual.shape[2] != target.shape[2]:
                B_p, T_p, N_p, D_p = pred_visual.shape
                N_t = target.shape[2]
                h_pred = int(N_p ** 0.5)
                h_tgt = int(N_t ** 0.5)
                pv = pred_visual.reshape(B_p * T_p, N_p, D_p)
                pv = pv.permute(0, 2, 1).reshape(B_p * T_p, D_p, h_pred, h_pred)
                pv = F.interpolate(pv, size=(h_tgt, h_tgt), mode="bilinear", align_corners=False)
                pv = pv.reshape(B_p * T_p, D_p, N_t).permute(0, 2, 1)
                pred_visual = pv.reshape(B_p, T_p, N_t, D_p)

            total_loss = total_loss + F.mse_loss(pred_visual, target)
            count += 1

        state["pred_deter"] = pred_deter

        if count == 0:
            return torch.tensor(0.0, device=self._device)
        return total_loss / count

    def kl_loss(self, post, prior, free=1.0, dyn_scale=0.5, rep_scale=0.1):
        """No KL loss for DINO-WM (deterministic predictor).

        Returns zeros for API compatibility.
        """
        zero = torch.tensor(0.0, device=self._device)
        return zero, zero, zero, zero
