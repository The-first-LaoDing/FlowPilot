"""3D Rotary Position Embedding for spatiotemporal video tokens."""

import torch
from typing import Tuple


def get_rotary_pos_embed(
    grid_sizes: list[int],
    head_dim: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 3D rotary position embedding frequencies.

    Splits head_dim into 3 even-sized segments for T, H, W axes so that
    rotation pairs never cross axis boundaries. Each of T and H gets
    floor(head_dim//3 / 2)*2 dimensions (rounded down to even); W gets
    the remainder.

    Args:
        grid_sizes: [T, H, W] grid dimensions.
        head_dim: Per-head dimension (e.g., 64).
        theta: RoPE base frequency.

    Returns:
        (freqs_cos, freqs_sin): each [T*H*W, head_dim].
    """
    T, H, W = grid_sizes

    # Round per-axis dims down to even so rotation pairs stay within one axis.
    dim_base = (head_dim // 3) // 2 * 2
    dims_per_axis = [dim_base, dim_base, head_dim - 2 * dim_base]

    all_angles = []
    for size, dim in zip(grid_sizes, dims_per_axis):
        # Frequency bands: dim//2 pairs, exponent denominator = dim (standard RoPE).
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        pos = torch.arange(size, dtype=torch.float32)
        # Outer product: [size, dim//2]; interleave to [size, dim]
        angles = torch.outer(pos, inv_freq)
        angles_expanded = torch.stack([angles, angles], dim=-1).flatten(-2)  # [size, dim]
        all_angles.append(angles_expanded)

    # Build 3D position grid and gather frequencies
    # Position indices: [T*H*W] for each axis
    t_idx = torch.arange(T).view(T, 1, 1).expand(T, H, W).reshape(-1)
    h_idx = torch.arange(H).view(1, H, 1).expand(T, H, W).reshape(-1)
    w_idx = torch.arange(W).view(1, 1, W).expand(T, H, W).reshape(-1)

    # Gather per-axis frequencies and concatenate
    angle_t = all_angles[0][t_idx]  # [T*H*W, dim_t]
    angle_h = all_angles[1][h_idx]  # [T*H*W, dim_h]
    angle_w = all_angles[2][w_idx]  # [T*H*W, dim_w]
    angles = torch.cat([angle_t, angle_h, angle_w], dim=-1)  # [T*H*W, head_dim]

    # Return cos and sin of angles
    return angles.cos(), angles.sin()


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to Q and K.

    Args:
        q: [B, S, N, D] query tensor.
        k: [B, S, N, D] key tensor.
        freqs: (freqs_cos, freqs_sin) each [S, D] from get_rotary_pos_embed.

    Returns:
        (q_rotated, k_rotated): same shapes as input.
    """
    freqs_cos, freqs_sin = freqs
    # Move to same device as q
    freqs_cos = freqs_cos.to(q.device, dtype=q.dtype)
    freqs_sin = freqs_sin.to(q.device, dtype=q.dtype)

    # Reshape for broadcasting: [1, S, 1, D]
    cos = freqs_cos.unsqueeze(0).unsqueeze(2)
    sin = freqs_sin.unsqueeze(0).unsqueeze(2)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        # Rotate pairs: [-x2, x1, -x4, x3, ...]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    q_rot = q * cos + rotate(q) * sin
    k_rot = k * cos + rotate(k) * sin
    return q_rot, k_rot
