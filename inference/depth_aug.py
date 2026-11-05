"""Depth clip + axial Gaussian noise + random dropout augmentation.

Pure numpy; only cv2 dependency is for the optional resize path.
"""

import cv2
import numpy as np

# Depth normalization basis shared by SimEnv and the model: depth values are
# in [0, 1] where 1.0 corresponds to _CAM_MAX_M metres.
_CAM_MAX_M = 20.0
_NET_H = 96
_NET_W = 160


def clip_and_noise_depth(
    depth01: np.ndarray,
    *,
    max_depth_m:   float,
    sigma0:        float = 0.0,
    k_quad:        float = 0.0,
    dropout_p:     float = 0.0,
    dropout_fill_m: float = None,
    resize_to_net: bool  = True,
) -> np.ndarray:
    """Clip to effective sensor range, add axial Gaussian noise, drop pixels.

    Noise model:  sigma(z) = sigma0 + k_quad * z^2   (metres).
    Dropout:      invalidated pixels are set to dropout_fill_m when provided,
                  otherwise hi (simulates SimEnv's NaN-to-max behaviour).

    Args:
        depth01:       [H, W] float32, normalized to [0, 1] (1.0 == 20 m).
        max_depth_m:   hard wall distance in metres (e.g. 10.0 for short-range).
        sigma0:        baseline noise std in metres (0.0 disables).
        k_quad:        quadratic noise coefficient (m / m^2). RealSense D435
                       ≈ 0.005 (50 cm std at 10 m). 0.0 disables.
        dropout_p:     per-pixel invalid-reading probability. 0.0 disables.
        dropout_fill_m: metre value for dropped pixels. None uses max_depth_m.
        resize_to_net: if True (rviz path), resize to (96, 160) before clipping;
                       input is typically native Flightmare 90x160. if False
                       (px4 path), input is expected to already be 96x160.

    Returns:
        When resize_to_net=True:  [_NET_H, _NET_W] float32.
        When resize_to_net=False: same shape as input (caller must ensure
            the input is already at the network size, e.g. 96x160).
        Values are normalized to [0, max_depth_m / _CAM_MAX_M].
    """
    if resize_to_net and depth01.shape != (_NET_H, _NET_W):
        depth01 = cv2.resize(depth01, (_NET_W, _NET_H),
                             interpolation=cv2.INTER_LINEAR)

    hi = float(max_depth_m) / _CAM_MAX_M
    out = np.minimum(depth01, hi).astype(np.float32, copy=True)

    if sigma0 > 0.0 or k_quad > 0.0:
        z_m = out * _CAM_MAX_M
        std_map_m = sigma0 + k_quad * (z_m * z_m)
        noise = (np.random.standard_normal(out.shape).astype(np.float32)
                 * (std_map_m / _CAM_MAX_M).astype(np.float32))
        out += noise
        np.clip(out, 0.0, hi, out=out)

    if dropout_p > 0.0:
        mask = np.random.random(out.shape) < dropout_p
        fill01 = hi if dropout_fill_m is None else float(dropout_fill_m) / _CAM_MAX_M
        out[mask] = np.clip(fill01, 0.0, hi)

    return out


def clip_and_noise_depth_from_cfg(
    depth01: np.ndarray,
    cfg,
    *,
    resize_to_net: bool = True,
) -> np.ndarray:
    """Convenience wrapper reading noise params from a cfg module/object.

    Required cfg field:  max_depth_m.
    Optional cfg fields: depth_noise_std_m, depth_noise_quad_coef,
                         depth_dropout_prob (default 0.0 if missing),
                         depth_dropout_fill_m (default max_depth_m).
    """
    return clip_and_noise_depth(
        depth01,
        max_depth_m=cfg.max_depth_m,
        sigma0=float(getattr(cfg, "depth_noise_std_m",     0.0)),
        k_quad=float(getattr(cfg, "depth_noise_quad_coef", 0.0)),
        dropout_p=float(getattr(cfg, "depth_dropout_prob", 0.0)),
        dropout_fill_m=getattr(cfg, "depth_dropout_fill_m", None),
        resize_to_net=resize_to_net,
    )
