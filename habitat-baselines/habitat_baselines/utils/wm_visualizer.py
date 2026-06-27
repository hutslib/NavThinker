"""
World Model visualization utilities for evaluation.

Generates visualization components for:
  - Depth: WM-reconstructed depth colormap
  - Trajectory: GT vs. WM-predicted human trajectories overlaid on top-down map

These components are composed into a unified eval video frame alongside
the standard RGB observation.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field


# Default GT future trajectory colors (RGB) — same hue as history trajectory in maps.py.
_DEFAULT_GT_COLORS = [
    (200, 50, 0),     # H1: dark red
    (0, 120, 80),     # H2: dark green
    (0, 80, 200),     # H3: dark blue
    (200, 140, 0),    # H4: dark orange
    (150, 0, 200),    # H5: dark magenta
    (0, 180, 180),    # H6: dark cyan
    (200, 0, 100),    # H7: red-violet
    (0, 150, 100),    # H8: teal
]

# Default Pred future trajectory colors (RGB) — brighter/lighter version of GT hue
_DEFAULT_PRED_COLORS = [
    (255, 120, 80),   # H1: bright red-orange
    (100, 255, 130),  # H2: bright green
    (60, 160, 255),   # H3: bright blue
    (255, 215, 0),    # H4: bright gold
    (220, 80, 255),   # H5: bright magenta
    (80, 255, 255),   # H6: bright cyan
    (255, 60, 180),   # H7: bright pink
    (80, 255, 150),   # H8: bright teal
]

_DEFAULT_ROBOT_GOAL_COLOR = (200, 0, 0)


def _get_traj_cfg_value(traj_cfg, key, default):
    """Read a value from traj_cfg (OmegaConf / dict / dataclass), fallback to default."""
    if traj_cfg is None:
        return default
    if hasattr(traj_cfg, key):
        return getattr(traj_cfg, key)
    if isinstance(traj_cfg, dict):
        return traj_cfg.get(key, default)
    return default


def _parse_color_list(raw, default):
    """Convert config color list (list of lists) to list of tuples."""
    if raw is None or raw is default:
        return default
    return [tuple(c) for c in raw]


@dataclass
class LookaheadActionResult:
    """Prediction result for a single candidate action."""
    action_id: int = -1
    action_name: str = ""
    pred_depth_raw: Optional[np.ndarray] = None   # (H, W) float32
    pred_depth_vis: Optional[np.ndarray] = None   # (H, W, 3) colormap
    depth_rmse: float = 0.0


@dataclass
class WMStepResult:
    """Per-step WM inference results for composing the unified eval frame."""
    pred_depth_vis: Optional[np.ndarray] = None   # (H, W, 3) colormap — best/taken action
    pred_depth_raw: Optional[np.ndarray] = None   # (H, W) float32 — best/taken action
    gt_traj: Optional[np.ndarray] = None           # (N, T, 2)
    pred_traj: Optional[np.ndarray] = None         # (N, T, 2)
    num_humans: int = 0
    human_positions: Optional[np.ndarray] = None   # (N, 2) current pos, robot-centric
    human_goals: Optional[np.ndarray] = None       # (N, 2) goal pos, robot-centric
    depth_rmse: float = 0.0
    traj_ade: float = 0.0
    lookahead_actions: Optional[List[LookaheadActionResult]] = None
    taken_action_id: int = -1


_COLORMAP_TABLE = {
    "AUTUMN": cv2.COLORMAP_AUTUMN,
    "BONE": cv2.COLORMAP_BONE,
    "JET": cv2.COLORMAP_JET,
    "WINTER": cv2.COLORMAP_WINTER,
    "RAINBOW": cv2.COLORMAP_RAINBOW,
    "OCEAN": cv2.COLORMAP_OCEAN,
    "SUMMER": cv2.COLORMAP_SUMMER,
    "SPRING": cv2.COLORMAP_SPRING,
    "COOL": cv2.COLORMAP_COOL,
    "HSV": cv2.COLORMAP_HSV,
    "PINK": cv2.COLORMAP_PINK,
    "HOT": cv2.COLORMAP_HOT,
    "PARULA": cv2.COLORMAP_PARULA,
    "MAGMA": cv2.COLORMAP_MAGMA,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "PLASMA": cv2.COLORMAP_PLASMA,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "CIVIDIS": cv2.COLORMAP_CIVIDIS,
    "TWILIGHT": cv2.COLORMAP_TWILIGHT,
    "TWILIGHT_SHIFTED": cv2.COLORMAP_TWILIGHT_SHIFTED,
    "TURBO": cv2.COLORMAP_TURBO,
    "DEEPGREEN": cv2.COLORMAP_DEEPGREEN,
}


def _resolve_colormap(name_or_int):
    """Accept a string name (e.g. 'TURBO') or cv2 int constant."""
    if isinstance(name_or_int, int):
        return name_or_int
    return _COLORMAP_TABLE.get(str(name_or_int).upper(), cv2.COLORMAP_TURBO)


def depth_to_colormap(depth, vmin=0.0, vmax=1.0, colormap=cv2.COLORMAP_TURBO):
    """Return an RGB colormap image from a single-channel depth array."""
    cmap = _resolve_colormap(colormap)
    depth_clipped = np.clip(depth, vmin, vmax)
    if vmax - vmin > 1e-6:
        depth_norm = ((depth_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth_clipped, dtype=np.uint8)
    bgr = cv2.applyColorMap(depth_norm, cmap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _alpha_overlay(base, overlay, alpha=0.55):
    """Blend *overlay* onto *base* with constant alpha (in-place on base)."""
    mask = np.any(overlay != 0, axis=-1)
    base[mask] = (
        base[mask].astype(np.float32) * (1 - alpha)
        + overlay[mask].astype(np.float32) * alpha
    ).astype(np.uint8)


def _draw_pin_marker(canvas, center, color, size=20):
    """Draw a map-pin (location marker) icon at *center* on *canvas*.

    The pin is an inverted teardrop: a filled circle on top with a triangular
    pointer at the bottom, plus a white inner circle to mimic the classic
    map-pin look.
    """
    cx, cy = int(center[0]), int(center[1])
    r = size // 2
    tip_y = cy + int(size * 0.95)

    tri_pts = np.array([
        [cx - int(r * 0.7), cy + int(r * 0.3)],
        [cx + int(r * 0.7), cy + int(r * 0.3)],
        [cx, tip_y],
    ], dtype=np.int32)
    cv2.fillConvexPoly(canvas, tri_pts, color, cv2.LINE_AA)

    cv2.circle(canvas, (cx, cy), r, color, -1, cv2.LINE_AA)

    inner_r = max(2, int(r * 0.42))
    cv2.circle(canvas, (cx, cy), inner_r, (255, 255, 255), -1, cv2.LINE_AA)


def render_depth_comparison(gt_depth, pred_depth, target_h=256, target_w=256):
    """Side-by-side GT vs predicted depth (used by trainer visualization).
    Returns BGR for cv2.imwrite compatibility."""
    if gt_depth.ndim == 3:
        gt_depth = gt_depth[..., 0]
    if pred_depth.ndim == 3:
        pred_depth = pred_depth[..., 0]

    gt_vis = depth_to_colormap(gt_depth)
    pred_vis = depth_to_colormap(pred_depth)
    gt_vis = cv2.resize(gt_vis, (target_w, target_h))
    pred_vis = cv2.resize(pred_vis, (target_w, target_h))

    gap = 4
    canvas = np.ones((target_h, 2 * target_w + gap, 3), dtype=np.uint8) * 255
    canvas[:, :target_w] = gt_vis
    canvas[:, target_w + gap:] = pred_vis

    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "GT Depth", (5, 20), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "WM Pred", (target_w + gap + 5, 20), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    rmse = float(np.sqrt(np.mean((gt_depth - pred_depth) ** 2)))
    cv2.putText(canvas, f"RMSE={rmse:.4f}", (target_w + gap + 5, target_h - 10),
                font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render_trajectory_comparison(
    gt_traj, pred_traj, num_humans,
    human_positions=None,
    canvas_size=512, world_range=8.0,
):
    """Standalone GT vs predicted trajectory plot (used by trainer visualization)."""
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    cx, cy = canvas_size // 2, canvas_size // 2
    scale = canvas_size / (2 * world_range)

    def world_to_px(pos):
        px = int(cx + pos[0] * scale)
        py = int(cy + pos[1] * scale)
        return (np.clip(px, 0, canvas_size - 1), np.clip(py, 0, canvas_size - 1))

    cv2.drawMarker(canvas, (cx, cy), (80, 80, 80), cv2.MARKER_DIAMOND, 14, 2)
    gt_layer = np.zeros_like(canvas)

    for h in range(min(num_humans, len(gt_traj), len(pred_traj))):
        gt_h, pred_h = gt_traj[h], pred_traj[h]
        if np.all(np.abs(gt_h) > 90):
            continue
        gt_color = _DEFAULT_GT_COLORS[h % len(_DEFAULT_GT_COLORS)]
        pd_color = _DEFAULT_PRED_COLORS[h % len(_DEFAULT_PRED_COLORS)]
        hp = None
        if human_positions is not None and h < len(human_positions):
            _hp = human_positions[h]
            if not np.all(np.abs(_hp) > 90):
                hp = _hp
        if len(pred_h) > 0:
            if hp is not None:
                cv2.line(canvas, world_to_px(hp), world_to_px(pred_h[0]), pd_color, 3, cv2.LINE_AA)
            for t in range(len(pred_h) - 1):
                cv2.line(canvas, world_to_px(pred_h[t]), world_to_px(pred_h[t + 1]), pd_color, 3, cv2.LINE_AA)
            cv2.circle(canvas, world_to_px(pred_h[0]), 6, pd_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, world_to_px(pred_h[-1]), 5, pd_color, 2, cv2.LINE_AA)
        if len(gt_h) > 0:
            if hp is not None:
                cv2.line(gt_layer, world_to_px(hp), world_to_px(gt_h[0]), gt_color, 3, cv2.LINE_AA)
            for t in range(len(gt_h) - 1):
                cv2.line(gt_layer, world_to_px(gt_h[t]), world_to_px(gt_h[t + 1]), gt_color, 3, cv2.LINE_AA)
            for t in range(len(gt_h)):
                pt = world_to_px(gt_h[t])
                r = 6 if t == 0 else (5 if t == len(gt_h) - 1 else 4)
                cv2.circle(gt_layer, pt, r, gt_color, -1, cv2.LINE_AA)

    _alpha_overlay(canvas, gt_layer, alpha=0.6)

    for h in range(min(num_humans, len(gt_traj), len(pred_traj))):
        if np.all(np.abs(gt_traj[h]) > 90):
            continue
        if human_positions is not None and h < len(human_positions):
            hp = human_positions[h]
            if not np.all(np.abs(hp) > 90):
                color = _DEFAULT_GT_COLORS[h % len(_DEFAULT_GT_COLORS)]
                cv2.drawMarker(canvas, world_to_px(hp), color, cv2.MARKER_CROSS, 14, 3, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (60, 60, 60)
    cv2.putText(canvas, "GT=transparent  Pred=opaque  X=CurPos", (5, 20),
                font, 0.42, text_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Humans: {num_humans}", (5, canvas_size - 10),
                font, 0.5, text_color, 1, cv2.LINE_AA)
    if num_humans > 0:
        valid = min(num_humans, len(gt_traj), len(pred_traj))
        ade_vals = []
        for h in range(valid):
            if np.all(np.abs(gt_traj[h]) > 90):
                continue
            ade_vals.append(np.linalg.norm(gt_traj[h] - pred_traj[h], axis=-1).mean())
        if ade_vals:
            cv2.putText(canvas, f"ADE={float(np.mean(ade_vals)):.3f}m",
                        (canvas_size - 160, canvas_size - 10), font, 0.5, text_color, 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def compose_wm_frame(*frames, target_h=256):
    """Horizontally stack any number of visualization panels."""
    panels = []
    for frame in frames:
        if frame is not None:
            h_ratio = target_h / frame.shape[0]
            panels.append(cv2.resize(frame, (int(frame.shape[1] * h_ratio), target_h)))
    if not panels:
        return None
    gap = 4
    total_w = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    canvas = np.ones((target_h, total_w, 3), dtype=np.uint8) * 255
    x = 0
    for p in panels:
        canvas[:, x:x + p.shape[1]] = p
        x += p.shape[1] + gap
    return canvas


_ACTION_NAMES = {0: "STOP", 1: "FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}


def render_action_icon(action_id: int, size: int = 48) -> np.ndarray:
    """Render an action as a small icon: arrow for movement, circle for stop."""
    icon = np.full((size, size, 3), 40, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    r = size // 3
    color = (0, 255, 0)

    if action_id == 0:
        cv2.circle(icon, (cx, cy), r, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(icon, (cx, cy), r // 3, (0, 0, 255), -1, cv2.LINE_AA)
    elif action_id == 1:
        cv2.arrowedLine(icon, (cx, cy + r), (cx, cy - r),
                        color, 2, cv2.LINE_AA, tipLength=0.35)
    elif action_id == 2:
        cv2.arrowedLine(icon, (cx + r, cy), (cx - r, cy),
                        (255, 200, 0), 2, cv2.LINE_AA, tipLength=0.35)
    elif action_id == 3:
        cv2.arrowedLine(icon, (cx - r, cy), (cx + r, cy),
                        (255, 100, 0), 2, cv2.LINE_AA, tipLength=0.35)
    else:
        cv2.putText(icon, "?", (cx - 8, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return icon


def render_rgb_observation(
    rgb_frames: list,
    time_indices: list,
    actions: list = None,
    tile_size: int = 192,
) -> np.ndarray:
    """Render RGB observation thumbnails in a horizontal strip.

    Args:
        rgb_frames: list of RGB arrays (H, W, 3), uint8 or float [0,1].
        time_indices: list of time step indices for labeling.
        actions: optional list of action ids for each frame.
        tile_size: size of each thumbnail.

    Returns:
        BGR image with a title bar and horizontally arranged RGB thumbnails.
    """
    imgs = []
    for idx, (rgb, t_idx) in enumerate(zip(rgb_frames, time_indices)):
        if rgb.dtype == np.float32 or rgb.dtype == np.float64:
            rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        vis = cv2.resize(rgb, (tile_size, tile_size))
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        act = actions[idx] if actions is not None and idx < len(actions) else None
        act_name = _ACTION_NAMES.get(act, "?") if act is not None else ""
        label = f"t={t_idx}"
        if act_name:
            label += f" ({act_name})"
        cv2.putText(vis_bgr, label, (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        if act is not None:
            icon = render_action_icon(act, size=24)
            vis_bgr[4:28, tile_size - 28:tile_size - 4] = icon

        imgs.append(vis_bgr)

    if not imgs:
        return None

    gap = 4
    total_w = sum(im.shape[1] for im in imgs) + gap * (len(imgs) - 1)
    img_h = imgs[0].shape[0]
    bar_h = 24
    canvas = np.full((bar_h + img_h, total_w, 3), 40, dtype=np.uint8)

    t_start, t_end = time_indices[0], time_indices[-1]
    cv2.putText(canvas, f"RGB Observation (t={t_start}..{t_end})",
                (8, bar_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    x = 0
    for im in imgs:
        canvas[bar_h:bar_h + im.shape[0], x:x + im.shape[1]] = im
        x += im.shape[1] + gap

    return canvas


def render_history_panel(
    depth_frames: list,
    actions: list,
    time_indices: list,
    tile_size: int = 192,
) -> np.ndarray:
    """Render GT history depth frames in a horizontal strip with action labels.

    Args:
        depth_frames: list of 2D depth arrays (H, W), values in [0, 1].
        actions: list of action ids (int or None) for each frame.
        time_indices: list of time step indices for labeling.
        tile_size: size of each depth thumbnail.

    Returns:
        BGR image with a title bar and horizontally arranged depth thumbnails.
    """
    hist_imgs = []
    for depth, act, t_idx in zip(depth_frames, actions, time_indices):
        vis = cv2.resize(depth_to_colormap(depth), (tile_size, tile_size))
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        act_name = _ACTION_NAMES.get(act, "?") if act is not None else ""
        label = f"t={t_idx}"
        if act_name:
            label += f" ({act_name})"
        cv2.putText(vis_bgr, label, (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        if act is not None:
            icon = render_action_icon(act, size=24)
            vis_bgr[4:28, tile_size - 28:tile_size - 4] = icon

        hist_imgs.append(vis_bgr)

    if not hist_imgs:
        return None

    gap = 4
    total_w = sum(im.shape[1] for im in hist_imgs) + gap * (len(hist_imgs) - 1)
    hist_h = hist_imgs[0].shape[0]
    bar_h = 24
    canvas = np.full((bar_h + hist_h, total_w, 3), 40, dtype=np.uint8)

    t_start, t_end = time_indices[0], time_indices[-1]
    cv2.putText(canvas, f"History (GT t={t_start}..{t_end})",
                (8, bar_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    x = 0
    for im in hist_imgs:
        canvas[bar_h:bar_h + im.shape[0], x:x + im.shape[1]] = im
        x += im.shape[1] + gap

    return canvas


def render_step_label(text: str, width: int = 768, height: int = 28,
                      action_id: int = None) -> np.ndarray:
    """Render a thin labeled divider bar, optionally with action info."""
    bar = np.full((height, width, 3), 40, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if action_id is not None:
        action_name = _ACTION_NAMES.get(int(action_id), f"ACT{action_id}")
        text = f"{text}  |  action: {action_name}"
        icon = render_action_icon(int(action_id), size=height - 2)
        x_icon = width - height - 4
        bar[1:1 + icon.shape[0], x_icon:x_icon + icon.shape[1]] = icon
    cv2.putText(bar, text, (8, height - 8), font, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def vstack_wm_panels(panels: list, target_w: int = 768, gap: int = 4):
    """Vertically stack visualization panels, scaling each to the same width.

    Mirrors the offline training visualization layout.
    """
    resized = []
    for p in panels:
        if p is None:
            continue
        w_ratio = target_w / p.shape[1]
        new_h = int(p.shape[0] * w_ratio)
        resized.append(cv2.resize(p, (target_w, new_h)))
    if not resized:
        return None
    total_h = sum(r.shape[0] for r in resized) + gap * (len(resized) - 1)
    canvas = np.ones((total_h, target_w, 3), dtype=np.uint8) * 255
    y = 0
    for r in resized:
        canvas[y:y + r.shape[0]] = r
        y += r.shape[0] + gap
    return canvas


def render_depth_triple(
    gt_depth: np.ndarray,
    dpt_from_gt: np.ndarray,
    dpt_from_pred: np.ndarray,
    target_h: int = 256,
    target_w: int = 256,
) -> np.ndarray:
    """Three-column panel: GT Depth | DPT(GT Feat) | DPT(Pred Feat).

    Shows the DPT depth decoder output from real vs predicted ViT features,
    matching the offline training visualization style.
    Returns BGR image for cv2.imwrite.
    """
    def _prep(d):
        if d.ndim == 3:
            d = d[..., 0]
        return cv2.resize(depth_to_colormap(d), (target_w, target_h))

    gt_vis = _prep(gt_depth)
    dpt_gt_vis = _prep(dpt_from_gt)
    dpt_pred_vis = _prep(dpt_from_pred)

    gap = 4
    w_total = 3 * target_w + 2 * gap
    canvas = np.ones((target_h, w_total, 3), dtype=np.uint8) * 255
    canvas[:, :target_w] = gt_vis
    canvas[:, target_w + gap:2 * target_w + gap] = dpt_gt_vis
    canvas[:, 2 * (target_w + gap):] = dpt_pred_vis

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas_bgr, "GT Depth", (5, 20), font, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas_bgr, "DPT(GT Feat)", (target_w + gap + 5, 20),
                font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas_bgr, "DPT(Pred Feat)", (2 * (target_w + gap) + 5, 20),
                font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    rmse_gt = float(np.sqrt(np.mean((gt_depth - dpt_from_gt) ** 2)))
    rmse_pred = float(np.sqrt(np.mean((gt_depth - dpt_from_pred) ** 2)))
    cv2.putText(canvas_bgr, f"RMSE={rmse_gt:.4f}",
                (target_w + gap + 5, target_h - 10), font, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas_bgr, f"RMSE={rmse_pred:.4f}",
                (2 * (target_w + gap) + 5, target_h - 10), font, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas_bgr


def render_cosine_sim_heatmap(
    gt_features: np.ndarray,
    pred_features: np.ndarray,
    spatial_h: int,
    spatial_w: int,
    target_h: int = 256,
    target_w: int = 256,
) -> np.ndarray:
    """Per-patch cosine similarity heatmap between GT and predicted features.

    Returns BGR image for cv2.imwrite.
    """
    dot = np.sum(gt_features * pred_features, axis=-1)
    gt_norm = np.linalg.norm(gt_features, axis=-1)
    pred_norm = np.linalg.norm(pred_features, axis=-1)
    cos_sim = dot / (gt_norm * pred_norm + 1e-8)

    cos_2d = cos_sim.reshape(spatial_h, spatial_w)
    cos_norm = ((cos_2d + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    cos_resized = cv2.resize(cos_norm, (target_w, target_h),
                             interpolation=cv2.INTER_NEAREST)
    heatmap = cv2.applyColorMap(cos_resized, cv2.COLORMAP_VIRIDIS)

    font = cv2.FONT_HERSHEY_SIMPLEX
    mean_sim = float(cos_sim.mean())
    min_sim = float(cos_sim.min())
    cv2.putText(heatmap, "Patch CosSim", (5, 20), font, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(heatmap, f"mean={mean_sim:.3f} min={min_sim:.3f}",
                (5, target_h - 10), font, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return heatmap


def draw_trajectories_on_topdown(
    topdown_map: np.ndarray,
    gt_traj: Optional[np.ndarray],
    pred_traj: Optional[np.ndarray],
    num_humans: int,
    human_positions: Optional[np.ndarray] = None,
    human_goals: Optional[np.ndarray] = None,
    robot_world_pos: Optional[np.ndarray] = None,
    goal_world_pos: Optional[np.ndarray] = None,
    bounds: Optional[Tuple] = None,
    map_shape: Optional[Tuple[int, int]] = None,
    ade: float = 0.0,
    traj_cfg=None,
):
    """Draw GT and predicted future trajectories on a colorized top-down map.

    Trajectories from sensors are in robot-centric 2D coords (delta_x, delta_z).
    We convert them to the same map-pixel coordinate system used by the
    TopDownMap measure so they align with the existing agent paths.

    Args:
        topdown_map: Already-colorized RGB top-down map (may be rotated/resized).
        gt_traj: (N, T, 2) GT future trajectory, robot-centric (x, z).
        pred_traj: (N, T, 2) predicted future trajectory, robot-centric (x, z).
        num_humans: number of valid humans.
        human_positions: (N, 2) current human positions, robot-centric (x, z).
        robot_world_pos: (3,) robot world position [x, y, z] from sim.
        bounds: (lower_bound, upper_bound) from pathfinder.get_bounds().
        map_shape: (H, W) of the *original* (pre-rotation, pre-resize) top-down map.
        ade: average displacement error to display.
    """
    canvas = topdown_map.copy()
    h, w = canvas.shape[:2]

    if gt_traj is None or pred_traj is None:
        return canvas

    has_map_info = (robot_world_pos is not None and bounds is not None
                    and map_shape is not None)

    if has_map_info:
        lower_bound, upper_bound = bounds
        raw_h, raw_w = map_shape  # original map before colorize/rotate/resize
        grid_size_z = abs(upper_bound[2] - lower_bound[2]) / raw_h
        grid_size_x = abs(upper_bound[0] - lower_bound[0]) / raw_w

        rotated = raw_h > raw_w

        if rotated:
            # After rot90: new shape = (raw_w, raw_h) then resized to (h, w)
            scale_x = w / raw_h
            scale_y = h / raw_w
        else:
            # No rotation: shape = (raw_h, raw_w) then resized to (h, w)
            scale_x = w / raw_w
            scale_y = h / raw_h

        def relative_to_px(delta):
            """Convert robot-centric (delta_x, delta_z) to pixel coords on the
            colorized (possibly rotated & resized) map.

            Sensor coords: delta = (world_x - robot_x, world_z - robot_z)
            to_grid maps: world_z -> grid_x (row), world_x -> grid_y (col)
            OpenCV draws at (col, row) = (grid_y, grid_x)
            rot90 CCW: (row, col) -> (orig_cols-1-col, row)
            """
            world_x = robot_world_pos[0] + delta[0]
            world_z = robot_world_pos[2] + delta[1]
            grid_row = (world_z - lower_bound[2]) / grid_size_z
            grid_col = (world_x - lower_bound[0]) / grid_size_x

            if rotated:
                # After rot90: new_row = orig_cols-1-col, new_col = row
                px = int(grid_row * scale_x)
                py = int((raw_w - 1 - grid_col) * scale_y)
            else:
                px = int(grid_col * scale_x)
                py = int(grid_row * scale_y)
            return (np.clip(px, 0, w - 1), np.clip(py, 0, h - 1))
    else:
        cx, cy = w // 2, h // 2
        scale = min(h, w) / 16.0

        def relative_to_px(delta):
            px = int(cx + delta[0] * scale)
            py = int(cy + delta[1] * scale)
            return (np.clip(px, 0, w - 1), np.clip(py, 0, h - 1))

    # ── Read configurable parameters from traj_cfg ──
    draw_gt = _get_traj_cfg_value(traj_cfg, "draw_gt_future", True)
    draw_pred = _get_traj_cfg_value(traj_cfg, "draw_pred_future", True)
    draw_human_goals_flag = _get_traj_cfg_value(traj_cfg, "draw_human_goals", True)
    draw_robot_goal_flag = _get_traj_cfg_value(traj_cfg, "draw_robot_goal", True)
    gt_alpha = float(_get_traj_cfg_value(traj_cfg, "gt_alpha", 0.35))
    pred_alpha = float(_get_traj_cfg_value(traj_cfg, "pred_alpha", 1.0))

    cfg_gt_thick = int(_get_traj_cfg_value(traj_cfg, "gt_thickness", 0))
    cfg_pred_thick = int(_get_traj_cfg_value(traj_cfg, "pred_thickness", 0))
    auto_thickness = max(1, min(h, w) // 150)
    gt_thickness = cfg_gt_thick if cfg_gt_thick > 0 else auto_thickness
    pred_thickness = cfg_pred_thick if cfg_pred_thick > 0 else auto_thickness

    gt_colors = _parse_color_list(
        _get_traj_cfg_value(traj_cfg, "gt_colors", None), _DEFAULT_GT_COLORS)
    pred_colors = _parse_color_list(
        _get_traj_cfg_value(traj_cfg, "pred_colors", None), _DEFAULT_PRED_COLORS)
    robot_goal_color = tuple(
        _get_traj_cfg_value(traj_cfg, "robot_goal_color", _DEFAULT_ROBOT_GOAL_COLOR))

    # Skip segments whose endpoints are too far apart in pixel space
    # (likely invalid data or extreme coordinate mismatch).
    _max_seg_px = max(h, w) * 0.4

    def _safe_line(target, p1, p2, color, thickness):
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        if dx > _max_seg_px or dy > _max_seg_px:
            return
        cv2.line(target, p1, p2, color, thickness, cv2.LINE_AA)

    # ── Draw predicted future trajectories ──
    if pred_alpha < 1.0:
        pred_layer = np.zeros_like(canvas)
    else:
        pred_layer = canvas

    if draw_pred:
        for hi in range(min(num_humans, len(gt_traj), len(pred_traj))):
            pred_h = pred_traj[hi]
            if np.all(np.abs(gt_traj[hi]) > 90):
                continue
            pd_color = pred_colors[hi % len(pred_colors)]

            hp = None
            if human_positions is not None and hi < len(human_positions):
                _hp = human_positions[hi]
                if not np.all(np.abs(_hp) > 90):
                    hp = _hp

            if len(pred_h) > 0:
                if hp is not None:
                    _safe_line(pred_layer, relative_to_px(hp), relative_to_px(pred_h[0]),
                               pd_color, pred_thickness)
                for t in range(len(pred_h) - 1):
                    _safe_line(pred_layer, relative_to_px(pred_h[t]),
                               relative_to_px(pred_h[t + 1]),
                               pd_color, pred_thickness)

    if draw_pred and pred_alpha < 1.0:
        _alpha_overlay(canvas, pred_layer, alpha=pred_alpha)

    # ── Draw GT future trajectories ──
    gt_layer = np.zeros_like(canvas)

    if draw_gt:
        for hi in range(min(num_humans, len(gt_traj), len(pred_traj))):
            gt_h = gt_traj[hi]
            if np.all(np.abs(gt_h) > 90):
                continue
            gt_color = gt_colors[hi % len(gt_colors)]

            hp = None
            if human_positions is not None and hi < len(human_positions):
                _hp = human_positions[hi]
                if not np.all(np.abs(_hp) > 90):
                    hp = _hp

            if len(gt_h) > 0:
                if hp is not None:
                    _safe_line(gt_layer, relative_to_px(hp), relative_to_px(gt_h[0]),
                               gt_color, gt_thickness)
                for t in range(len(gt_h) - 1):
                    _safe_line(gt_layer, relative_to_px(gt_h[t]),
                               relative_to_px(gt_h[t + 1]),
                               gt_color, gt_thickness)

    if draw_gt:
        _alpha_overlay(canvas, gt_layer, alpha=gt_alpha)

    # ── Markers ──
    marker_size = max(10, min(h, w) // 30)

    if draw_human_goals_flag:
        for hi in range(min(num_humans, len(gt_traj), len(pred_traj))):
            if np.all(np.abs(gt_traj[hi]) > 90):
                continue
            color = gt_colors[hi % len(gt_colors)]
            if human_goals is not None and hi < len(human_goals):
                goal = human_goals[hi]
                if not np.all(np.abs(goal) > 90):
                    cv2.drawMarker(canvas, relative_to_px(goal), color,
                                   cv2.MARKER_STAR, marker_size, 2, cv2.LINE_AA)

    if draw_robot_goal_flag and goal_world_pos is not None and has_map_info:
        goal_rel = np.array([
            goal_world_pos[0] - robot_world_pos[0],
            goal_world_pos[2] - robot_world_pos[2],
        ])
        _draw_pin_marker(canvas, relative_to_px(goal_rel), robot_goal_color,
                         size=max(14, min(h, w) // 20))

    return canvas


def _render_lookahead_strip(
    lookahead_actions: List[LookaheadActionResult],
    taken_action_id: int,
    target_h: int,
    colormap=cv2.COLORMAP_TURBO,
) -> Optional[np.ndarray]:
    """Render a horizontal strip of depth predictions for all candidate actions.

    Layout per action tile:
        ┌─────────────────┐
        │  action label    │  ← 20px label bar
        │                  │
        │  predicted depth │  ← depth colormap
        │                  │
        │  RMSE=0.1234     │  ← bottom label
        └─────────────────┘

    The tile for the taken action gets a green border highlight.
    """
    if not lookahead_actions:
        return None

    tile_h = target_h
    tile_w = tile_h
    label_h = 22
    gap = 3
    border = 3
    font = cv2.FONT_HERSHEY_SIMPLEX

    _TAKEN_COLOR = (0, 220, 80)
    _BORDER_COLOR = (80, 80, 80)

    tiles = []
    for la in lookahead_actions:
        tile = np.full((tile_h, tile_w, 3), 30, dtype=np.uint8)

        if la.pred_depth_vis is not None:
            cmap = _resolve_colormap(colormap)
            if la.pred_depth_raw is not None:
                vis = depth_to_colormap(la.pred_depth_raw, colormap=cmap)
            else:
                vis = la.pred_depth_vis
            img_h = tile_h - 2 * label_h
            vis_resized = cv2.resize(vis, (tile_w - 2 * border, img_h))
            tile[label_h:label_h + img_h, border:tile_w - border] = vis_resized

        is_taken = la.action_id == taken_action_id
        label_color = _TAKEN_COLOR if is_taken else (200, 200, 200)
        marker = " *" if is_taken else ""
        cv2.putText(tile, f"{la.action_name}{marker}", (4, label_h - 5),
                    font, 0.42, label_color, 1, cv2.LINE_AA)

        if la.action_id >= 0:
            icon = render_action_icon(la.action_id, size=label_h - 4)
            ix = tile_w - label_h
            tile[2:2 + icon.shape[0], ix:ix + icon.shape[1]] = icon

        rmse_text = f"RMSE={la.depth_rmse:.4f}" if la.depth_rmse > 0 else ""
        cv2.putText(tile, rmse_text, (4, tile_h - 6),
                    font, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        if is_taken:
            cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1),
                          _TAKEN_COLOR, border, cv2.LINE_AA)
        else:
            cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1),
                          _BORDER_COLOR, 1)

        tiles.append(tile)

    total_w = sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1)
    strip_label_h = 22
    strip = np.full((strip_label_h + tile_h, total_w, 3), 30, dtype=np.uint8)
    cv2.putText(strip, "WM Lookahead (1-step imagination per action)",
                (6, strip_label_h - 5), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    x = 0
    for t in tiles:
        strip[strip_label_h:strip_label_h + t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1] + gap

    return strip


def compose_unified_frame(
    rgb_obs: np.ndarray,
    gt_depth_obs: np.ndarray,
    pred_depth_vis: Optional[np.ndarray],
    topdown_map: Optional[np.ndarray],
    wm_result: Optional[WMStepResult] = None,
    depth_rmse: float = 0.0,
    robot_world_pos: Optional[np.ndarray] = None,
    goal_world_pos: Optional[np.ndarray] = None,
    bounds: Optional[Tuple] = None,
    raw_map_shape: Optional[Tuple[int, int]] = None,
    traj_cfg=None,
    third_rgb: Optional[np.ndarray] = None,
):
    """Build the unified eval frame.

    Top row:  [metrics bar | RGB | GT Depth | Pred Depth | Third RGB | TopDown+Traj]
    Bottom row (if lookahead available):
              [STOP pred | FORWARD pred | TURN_LEFT pred | TURN_RIGHT pred]

    Returns RGB uint8.
    """
    depth_cmap = _resolve_colormap(
        _get_traj_cfg_value(traj_cfg, "depth_colormap", "TURBO"))

    if rgb_obs is not None:
        target_h = rgb_obs.shape[0]
    elif gt_depth_obs is not None:
        d = gt_depth_obs
        if not isinstance(d, np.ndarray):
            d = np.asarray(d)
        target_h = d.shape[0]
    else:
        target_h = 256
    panels: List[np.ndarray] = []

    if rgb_obs is not None:
        panels.append(rgb_obs)

    if gt_depth_obs is not None:
        gt_depth_vis = _depth_obs_to_vis(gt_depth_obs, target_h, colormap=depth_cmap)
        panels.append(gt_depth_vis)

    if third_rgb is not None:
        t_rgb = third_rgb
        if not isinstance(t_rgb, np.ndarray):
            t_rgb = np.asarray(t_rgb)
        if t_rgb.dtype != np.uint8:
            t_rgb = np.clip(t_rgb * 255, 0, 255).astype(np.uint8)
        h_ratio = target_h / t_rgb.shape[0]
        new_w = int(t_rgb.shape[1] * h_ratio)
        t_rgb = cv2.resize(t_rgb, (new_w, target_h))
        panels.append(t_rgb)

    if topdown_map is not None:
        td = topdown_map.copy()
        if wm_result is not None and wm_result.gt_traj is not None:
            td = draw_trajectories_on_topdown(
                td,
                wm_result.gt_traj, wm_result.pred_traj,
                wm_result.num_humans,
                human_positions=wm_result.human_positions,
                human_goals=wm_result.human_goals,
                robot_world_pos=robot_world_pos,
                goal_world_pos=goal_world_pos,
                bounds=bounds,
                map_shape=raw_map_shape,
                ade=wm_result.traj_ade,
                traj_cfg=traj_cfg,
            )
        h_ratio = target_h / td.shape[0]
        new_w = int(td.shape[1] * h_ratio)
        td = cv2.resize(td, (new_w, target_h))
        panels.append(td)

    gap = 2
    top_row_w = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)

    # ── Build metrics text bar above panels ──
    overlay_cfg = _get_traj_cfg_value(traj_cfg, "overlay_text", None)
    show_text = bool(_get_traj_cfg_value(overlay_cfg, "show", True))

    bar_h = 0
    bar = None
    if show_text:
        bar_h = int(_get_traj_cfg_value(overlay_cfg, "bar_height", 28))
        font_scale = float(_get_traj_cfg_value(overlay_cfg, "font_scale", 0.5))
        text_color = tuple(_get_traj_cfg_value(overlay_cfg, "color", (255, 255, 255)))
        bg_color = tuple(_get_traj_cfg_value(overlay_cfg, "bg_color", (40, 40, 40)))
        text_thick = int(_get_traj_cfg_value(overlay_cfg, "thickness", 1))

        parts: List[str] = []
        if wm_result is not None:
            if depth_rmse > 0:
                parts.append(f"Depth RMSE: {depth_rmse:.4f}")
            if wm_result.traj_ade > 0:
                parts.append(f"Traj ADE: {wm_result.traj_ade:.3f}m")
            parts.append(f"Humans: {wm_result.num_humans}")
            if wm_result.taken_action_id >= 0:
                act_name = _ACTION_NAMES.get(wm_result.taken_action_id, "?")
                parts.append(f"Action: {act_name}")

    lookahead_strip = None
    if wm_result is not None and wm_result.lookahead_actions:
        _keep_ids = {1, 2, 3}  # FORWARD, TURN_LEFT, TURN_RIGHT
        filtered_la = [la for la in wm_result.lookahead_actions
                       if la.action_id in _keep_ids]
        if filtered_la:
            lookahead_strip = _render_lookahead_strip(
                filtered_la,
                wm_result.taken_action_id,
                target_h=target_h,
                colormap=depth_cmap,
            )

    # ── Determine final canvas width ──
    bottom_row_w = lookahead_strip.shape[1] if lookahead_strip is not None else 0
    total_w = max(top_row_w, bottom_row_w)

    if show_text and parts:
        bar = np.full((bar_h, total_w, 3), bg_color, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "  |  ".join(parts)
        text_y = int(bar_h * 0.7)
        cv2.putText(bar, text, (6, text_y), font, font_scale,
                    text_color, text_thick, cv2.LINE_AA)

    bottom_h = lookahead_strip.shape[0] if lookahead_strip is not None else 0
    canvas_h = bar_h + target_h + bottom_h
    canvas = np.zeros((canvas_h, total_w, 3), dtype=np.uint8)

    if bar is not None:
        canvas[:bar_h, :bar.shape[1]] = bar

    x = 0
    for p in panels:
        canvas[bar_h:bar_h + target_h, x:x + p.shape[1]] = p
        x += p.shape[1] + gap

    if lookahead_strip is not None:
        y_off = bar_h + target_h
        lw = min(lookahead_strip.shape[1], total_w)
        canvas[y_off:y_off + lookahead_strip.shape[0], :lw] = lookahead_strip[:, :lw]

    return canvas


def _depth_obs_to_vis(depth_obs: np.ndarray, target_h: int, colormap=cv2.COLORMAP_TURBO) -> np.ndarray:
    """Convert a raw depth observation (H,W,1) or (H,W) to a colormapped RGB image."""
    d = depth_obs
    if not isinstance(d, np.ndarray):
        d = d.cpu().numpy()
    if d.dtype == np.uint8:
        d = d.astype(np.float32) / 255.0
    if d.ndim == 3:
        d = d[..., 0]
    vis = depth_to_colormap(d, colormap=colormap)
    vis = cv2.resize(vis, (target_h, target_h))
    return vis


class WMVisualizer:
    """Stateful helper that runs WM lookahead prediction each eval step.

    At each step the visualizer produces a **prior prediction** of the
    current frame — i.e. what the WM *expected* to see given only the
    previous observations and the action that was taken.  This is the
    true measure of the WM's predictive (lookahead) ability.

    Timeline (called once per env step, after ``envs.step``):

        The evaluator loop looks like:

            batch = obs_t                          # loop top
            act_t = policy(obs_t)                  # prev_actions updated to act_t
            obs_{t+1} = envs.step(act_t)           # env transition
            batch = obs_{t+1}                      # batch updated
            wm_vis.step(batch, prev_actions, masks) # prev_actions = act_t

        So when ``step`` is called:
          - ``batch``        = obs_{t+1}  (the new observation)
          - ``prev_actions`` = act_t      (action that caused the transition)
          - ``masks``        = not-done   (False if episode just ended)

        What we do:
          1. **Prior prediction**: use ``img_step_lookahead(state_t, act_t)``
             to predict obs_{t+1}'s features *without seeing obs_{t+1}*.
             Decode depth / trajectory → WMStepResult.

          2. **History update**: encode the *real* obs_{t+1} and call
             ``observe_post_action(embed_{t+1}, act_t)`` to append to the
             history buffer.  We use act_t (not act_{t+1}) because the
             DINO-WM convention stores (obs, action_taken_at_obs) pairs,
             and at this point act_t is the action that was taken *at*
             obs_{t+1}'s predecessor.  However, the action truly associated
             with obs_{t+1} is act_{t+1} which hasn't been chosen yet.
             We use act_t as a placeholder; the slight mismatch on the
             action token is acceptable for visualization purposes since
             the visual patch tokens (which dominate the representation)
             are ground truth.

        The first step of each episode has no prior to predict from, so
        it falls back to GT-feature decoding (reconstruction).
    """

    def __init__(self, world_model, device, num_envs=1):
        self.world_model = world_model
        self.device = device
        self.num_envs = num_envs
        self._prior_states = [None] * num_envs
        self._vis_hist_z = [None] * num_envs
        self._step_count = 0
        self._env_step_counts = [0] * num_envs
        self._is_dino_wm = hasattr(world_model.dynamics, 'observe_post_action')

    def reset_env(self, env_idx):
        self._prior_states[env_idx] = None
        self._vis_hist_z[env_idx] = None
        self._env_step_counts[env_idx] = 0

    @torch.no_grad()
    def step(self, batch, prev_actions, masks, env_idx) -> Optional[WMStepResult]:
        """Run one WM visualisation step.

        Args:
            batch: current observations (obs_{t+1}, after envs.step).
            prev_actions: action that *led to* this observation (act_t).
            masks: not-done masks (False at episode boundary).
            env_idx: environment index.
        """
        wm = self.world_model
        if wm is None:
            return None

        _log = self._step_count < 3
        self._step_count += 1
        is_new_episode = not masks[env_idx].any().item()

        if is_new_episode:
            self.reset_env(env_idx)

        obs_i = {}
        for k, v in batch.items():
            val = v[env_idx:env_idx + 1]
            obs_i[k] = val
            stripped = k.replace("agent_0_", "", 1) if k.startswith("agent_0_") else None
            if stripped is not None:
                obs_i[stripped] = val

        if _log:
            print(f"[WMVis] obs keys: {list(obs_i.keys())}")

        gt_embed = wm.encoder(obs_i)
        if _log:
            print(f"[WMVis] gt_embed: {tuple(gt_embed.shape)}")

        # ── Action encoding ──
        action_i = prev_actions[env_idx:env_idx + 1]
        if action_i.dim() > 1:
            action_i = action_i.squeeze(-1)
        num_actions = wm.dynamics._num_actions
        if action_i.dtype in (torch.long, torch.int32, torch.int64):
            taken_action_id = int(action_i.clamp(min=0).item())
            action_oh = F.one_hot(action_i.clamp(min=0), num_classes=num_actions).float()
        else:
            taken_action_id = int(action_i.argmax(-1).item()) if action_i.dim() > 0 else 0
            action_oh = action_i.unsqueeze(0) if action_i.dim() == 0 else action_i

        is_first_step = self._env_step_counts[env_idx] == 0
        result = WMStepResult()
        result.taken_action_id = taken_action_id

        # ── Prepare GT depth for RMSE computation ──
        depth_key = next((k for k in obs_i if "depth" in k.lower()), None)
        gt_depth_np = None
        if depth_key is not None:
            gt_depth_raw = obs_i[depth_key][0].cpu().numpy()
            if gt_depth_raw.max() > 1.0:
                gt_depth_raw = gt_depth_raw / 255.0
            gt_depth_np = gt_depth_raw[..., 0] if gt_depth_raw.ndim == 3 else gt_depth_raw

        has_depth_head = "depth" in wm.heads
        depth_head = wm.heads["depth"] if has_depth_head else None
        uses_patches = hasattr(depth_head, '_h') if depth_head is not None else False

        def _decode_depth(patch_feat):
            """Decode depth from patch features, return (raw_np, vis_np, rmse)."""
            if depth_head is None or gt_depth_np is None:
                return None, None, 0.0
            if uses_patches:
                dist = depth_head(patch_feat)
            else:
                dist = depth_head(patch_feat.mean(dim=-2))
            raw = dist.mean()[0].cpu().numpy()
            if raw.ndim == 3:
                raw = raw[..., 0]
            if gt_depth_np.shape != raw.shape:
                raw = cv2.resize(raw, (gt_depth_np.shape[1], gt_depth_np.shape[0]))
            vis = depth_to_colormap(raw)
            rmse = float(np.sqrt(np.mean((gt_depth_np - raw) ** 2)))
            return raw.copy(), vis, rmse

        # ── 1. Prior prediction for the taken action (main panel) ──
        # Swap in per-env history buffer so we don't collide with the
        # main eval loop's _hist_z (which has batch=num_envs).
        saved_global_hist = wm.dynamics._hist_z if self._is_dino_wm else None
        if self._is_dino_wm:
            wm.dynamics._hist_z = self._vis_hist_z[env_idx]

        gt_patch_feat = getattr(wm.encoder, '_cached_vit_patch_features', None)
        if is_first_step:
            pred_patch_feat = gt_patch_feat if gt_patch_feat is not None else gt_embed
            if _log:
                print(f"[WMVis] first step: using {'GT patch feat' if gt_patch_feat is not None else 'GT embed'} "
                      f"(no prior to predict from), shape={tuple(pred_patch_feat.shape)}")
        else:
            prior_state = self._prior_states[env_idx]
            if prior_state is None:
                prior_state = wm.dynamics.initial(1)
            imagined = wm.dynamics.img_step_lookahead(prior_state, action_oh)
            pred_patch_feat = imagined["deter"]  # (1, N, D)
            if _log:
                print(f"[WMVis] prior prediction: {tuple(pred_patch_feat.shape)}")

        pred_feat_pooled = pred_patch_feat.mean(dim=-2)  # (1, D)

        # ── 2. Decode depth for the taken action ──
        raw, vis, rmse = _decode_depth(pred_patch_feat)
        if raw is not None:
            result.pred_depth_raw = raw
            result.pred_depth_vis = vis
            result.depth_rmse = rmse

        # ── 3. Lookahead for ALL candidate actions ──
        if not is_first_step and self._is_dino_wm:
            prior_state = self._prior_states[env_idx]
            if prior_state is None:
                prior_state = wm.dynamics.initial(1)
            lookahead_results = []
            for aid in range(num_actions):
                a_oh = F.one_hot(
                    torch.tensor([aid], device=self.device), num_classes=num_actions,
                ).float()
                imag = wm.dynamics.img_step_lookahead(prior_state, a_oh)
                a_raw, a_vis, a_rmse = _decode_depth(imag["deter"])
                lookahead_results.append(LookaheadActionResult(
                    action_id=aid,
                    action_name=_ACTION_NAMES.get(aid, f"ACT{aid}"),
                    pred_depth_raw=a_raw,
                    pred_depth_vis=a_vis,
                    depth_rmse=a_rmse,
                ))
            result.lookahead_actions = lookahead_results

        # ── 4. Decode trajectory from prior-predicted features ──
        traj_key = next(
            (k for k in obs_i if "future_trajectory" in k.lower()), None
        )
        if traj_key is not None and "human_traj" in wm.heads:
            human_state_goal_key = next(
                (k for k in obs_i if "human_state_goal" in k), None
            )
            human_state_goal = obs_i[human_state_goal_key] if human_state_goal_key else None
            if human_state_goal is not None:
                human_state_goal = human_state_goal.unsqueeze(1)

            feat_for_traj = pred_feat_pooled.unsqueeze(1)  # (1, 1, D)
            traj_dist = wm.heads["human_traj"](
                feat_for_traj, human_state_goal=human_state_goal,
            )
            pred_traj = traj_dist.mean()

            gt_traj_raw = obs_i[traj_key][0].cpu().numpy()
            pred_traj_np = pred_traj[0, 0].cpu().numpy()

            human_num_key = next(
                (k for k in obs_i if "human_num" in k.lower()), None
            )
            if human_num_key is not None:
                num_humans = int(obs_i[human_num_key][0].item())
            else:
                num_humans = gt_traj_raw.shape[0]

            human_positions = None
            human_goals = None
            if human_state_goal_key is not None:
                hsg_np = obs_i[human_state_goal_key][0].cpu().numpy()
                human_positions = hsg_np[:, :2]
                human_goals = hsg_np[:, 4:6]

            result.gt_traj = gt_traj_raw
            result.pred_traj = pred_traj_np
            result.num_humans = num_humans
            result.human_positions = human_positions
            result.human_goals = human_goals

            valid = min(num_humans, len(gt_traj_raw), len(pred_traj_np))
            ade_vals = []
            for hi in range(valid):
                if np.all(np.abs(gt_traj_raw[hi]) > 90):
                    continue
                disp = np.linalg.norm(
                    gt_traj_raw[hi] - pred_traj_np[hi], axis=-1,
                )
                ade_vals.append(disp.mean())
            if ade_vals:
                result.traj_ade = float(np.mean(ade_vals))

        # ── 4. Update history with GT obs_{t+1} so next step can predict from it ──
        is_first_flag = torch.zeros(1, 1, device=self.device)
        if is_new_episode:
            is_first_flag[:] = 1.0

        if self._is_dino_wm:
            patch_for_hist = gt_patch_feat if gt_patch_feat is not None else gt_embed
            wm.dynamics.observe_post_action(patch_for_hist, action_oh, is_first_flag)
            self._prior_states[env_idx] = {"deter": patch_for_hist}
            # Save per-env buffer and restore global buffer
            self._vis_hist_z[env_idx] = wm.dynamics._hist_z
            wm.dynamics._hist_z = saved_global_hist
        else:
            embed_seq = gt_embed.unsqueeze(1)
            action_seq = action_oh.unsqueeze(1)
            post, _ = wm.dynamics.observe(
                embed_seq, action_seq, is_first_flag,
                self._prior_states[env_idx],
            )
            self._prior_states[env_idx] = {
                k: v[:, 0] for k, v in post.items()
            }

        self._env_step_counts[env_idx] += 1

        return result


# ---------------------------------------------------------------------------
# Feature PCA visualization (probing – not used in training)
# ---------------------------------------------------------------------------

def render_feature_pca_comparison(
    gt_features: np.ndarray,
    pred_features: np.ndarray,
    spatial_h: int,
    spatial_w: int,
    target_h: int = 256,
    target_w: int = 256,
) -> np.ndarray:
    """Side-by-side PCA visualization of GT vs predicted ViT patch features.

    Projects both feature maps to the same 3 principal components (fitted on
    their concatenation), reshapes to a spatial grid, and renders as RGB.
    Also reports per-patch cosine similarity.

    Args:
        gt_features: (N_patches, D) real ViT patch features.
        pred_features: (N_patches, D) predicted ViT patch features.
        spatial_h, spatial_w: spatial grid size for reshaping.
        target_h, target_w: output panel size.

    Returns:
        BGR image with GT-PCA | Pred-PCA side by side.
    """
    both = np.concatenate([gt_features, pred_features], axis=0).astype(np.float32)
    mean = both.mean(axis=0, keepdims=True)
    centered = both - mean
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top3 = eigvecs[:, -3:][:, ::-1]  # largest 3 eigenvectors
    both_3 = centered @ top3  # (2N, 3)

    n = gt_features.shape[0]
    gt_3, pred_3 = both_3[:n], both_3[n:]

    def _to_rgb(arr, h, w):
        vmin, vmax = arr.min(), arr.max()
        if vmax - vmin > 1e-8:
            arr = (arr - vmin) / (vmax - vmin)
        else:
            arr = np.zeros_like(arr)
        img = arr.reshape(h, w, 3)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        return cv2.resize(img, (target_w, target_h))

    gt_rgb = _to_rgb(gt_3, spatial_h, spatial_w)
    pred_rgb = _to_rgb(pred_3, spatial_h, spatial_w)

    gap = 4
    canvas = np.ones((target_h, 2 * target_w + gap, 3), dtype=np.uint8) * 255
    canvas[:, :target_w] = gt_rgb
    canvas[:, target_w + gap:] = pred_rgb

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas_bgr, "GT ViT Feat (PCA)", (5, 20), font, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas_bgr, "Pred ViT Feat (PCA)", (target_w + gap + 5, 20),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    dot = np.sum(gt_features * pred_features, axis=-1)
    norms = np.linalg.norm(gt_features, axis=-1) * np.linalg.norm(pred_features, axis=-1) + 1e-8
    cos_sim = float(np.mean(dot / norms))
    mse = float(np.mean((gt_features - pred_features) ** 2))
    cv2.putText(canvas_bgr, f"CosSim={cos_sim:.3f}  MSE={mse:.4f}",
                (target_w + gap + 5, target_h - 10), font, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas_bgr


# ---------------------------------------------------------------------------
# Linear Depth Probe (frozen, BNHead-style – not used in training)
# ---------------------------------------------------------------------------

class LinearDepthProbe:
    """Lightweight linear probe: ViT patch features → depth map.

    Mimics DINOv2/DA BNHead: reshape patches to 2D, apply a learned 1×1 conv
    (D_vit → 1), bilinear upsample to target resolution.

    The probe is fitted on (gt_vit_features, gt_depth) pairs from the current
    batch via least-squares, then used to decode predicted ViT features.
    No GPU parameters; pure numpy for simplicity.
    """

    def __init__(self):
        self.weight = None   # (D_vit,)
        self.bias = 0.0

    def fit(self, features: np.ndarray, depth: np.ndarray):
        """Fit linear mapping: features @ w + b ≈ depth (per-patch average).

        Args:
            features: (N_patches, D_vit) float32
            depth: (H_depth, W_depth) float32, values in [0, 1]
        """
        N, D = features.shape
        h = int(N ** 0.5)
        w = h if h * h == N else N // h

        depth_resized = cv2.resize(depth, (w, h)).reshape(N)

        # Least-squares: features @ w + b ≈ depth_resized
        # Augment features with bias column
        X = np.concatenate([features, np.ones((N, 1), dtype=np.float32)], axis=1)
        # Solve via pseudo-inverse (numerically stable)
        sol, _, _, _ = np.linalg.lstsq(X, depth_resized, rcond=None)
        self.weight = sol[:-1]
        self.bias = sol[-1]

    def predict(self, features: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Predict depth map from patch features.

        Args:
            features: (N_patches, D_vit) float32
            target_h, target_w: output depth map resolution.

        Returns:
            depth_map: (target_h, target_w) float32
        """
        if self.weight is None:
            return np.zeros((target_h, target_w), dtype=np.float32)
        N = features.shape[0]
        h = int(N ** 0.5)
        w = h if h * h == N else N // h
        depth_patch = features @ self.weight + self.bias
        depth_2d = depth_patch.reshape(h, w)
        return cv2.resize(depth_2d, (target_w, target_h)).astype(np.float32)

    @property
    def is_fitted(self):
        return self.weight is not None


def render_probe_depth_comparison(
    gt_depth: np.ndarray,
    probe_from_gt: np.ndarray,
    probe_from_pred: np.ndarray,
    target_h: int = 256,
    target_w: int = 256,
) -> np.ndarray:
    """Three-column panel: GT Depth | Probe(GT ViT) | Probe(Pred ViT).

    Shows what the linear probe produces from real vs predicted ViT features,
    compared to the actual ground-truth depth.

    Returns BGR image for cv2.imwrite.
    """
    def _prep(d):
        if d.ndim == 3:
            d = d[..., 0]
        return cv2.resize(depth_to_colormap(d), (target_w, target_h))

    gt_vis = _prep(gt_depth)
    probe_gt_vis = _prep(probe_from_gt)
    probe_pred_vis = _prep(probe_from_pred)

    gap = 4
    w_total = 3 * target_w + 2 * gap
    canvas = np.ones((target_h, w_total, 3), dtype=np.uint8) * 255
    canvas[:, :target_w] = gt_vis
    canvas[:, target_w + gap:2 * target_w + gap] = probe_gt_vis
    canvas[:, 2 * (target_w + gap):] = probe_pred_vis

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas_bgr, "GT Depth", (5, 20), font, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas_bgr, "Probe(GT ViT)", (target_w + gap + 5, 20),
                font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas_bgr, "Probe(Pred ViT)", (2 * (target_w + gap) + 5, 20),
                font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    rmse_gt = float(np.sqrt(np.mean((gt_depth - probe_from_gt) ** 2)))
    rmse_pred = float(np.sqrt(np.mean((gt_depth - probe_from_pred) ** 2)))
    cv2.putText(canvas_bgr, f"RMSE={rmse_gt:.4f}",
                (target_w + gap + 5, target_h - 10), font, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas_bgr, f"RMSE={rmse_pred:.4f}",
                (2 * (target_w + gap) + 5, target_h - 10), font, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas_bgr
