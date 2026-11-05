"""Timestep embedding utilities for Flow Matching."""

import math
import torch
import torch.nn as nn


def sinusoidal_embedding_1d(
    timesteps: torch.Tensor,
    dim: int,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """Sinusoidal positional embedding for scalar timesteps.

    Args:
        timesteps: [B] float tensor.
        dim: Embedding dimension.
        max_period: Controls frequency range.

    Returns:
        [B, dim] embedding.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class VideoTimestepEmbedder(nn.Module):
    """Video stream timestep embedder.

    Maps integer t_id → (time_emb_v [B,D], adaln_params_v [B,6,D]).
    """

    def __init__(self, d_model: int = 512, d_embed: int = 256):
        super().__init__()
        self.d_model = d_model
        self.d_embed = d_embed
        # t → sinusoidal(d_embed) → MLP → d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_embed, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # d_model → 6*d_model (AdaLN params)
        self.adaln_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )

    def forward(self, t: torch.Tensor):
        """
        Args:
            t: [B] integer timestep ids.
        Returns:
            time_emb: [B, d_model] — used by output head.
            adaln_params: [B, 6, d_model] — per-layer AdaLN modulation.
        """
        emb = sinusoidal_embedding_1d(t.float(), self.d_embed)  # [B, d_embed]
        time_emb = self.mlp(emb)  # [B, d_model]
        adaln_flat = self.adaln_proj(time_emb)  # [B, 6*d_model]
        adaln_params = adaln_flat.view(-1, 6, self.d_model)  # [B, 6, d_model]
        return time_emb, adaln_params


class ActionTimestepEmbedder(nn.Module):
    """Action stream timestep embedder.

    Maps integer t_id → (time_emb_a [B,D], adaln_params_a [B,6,D]).
    """

    def __init__(self, d_model: int = 256, d_embed: int = 256):
        super().__init__()
        self.d_model = d_model
        self.d_embed = d_embed
        self.mlp = nn.Sequential(
            nn.Linear(d_embed, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.adaln_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )

    def forward(self, t: torch.Tensor):
        """
        Args:
            t: [B] integer timestep ids.
        Returns:
            time_emb: [B, d_model].
            adaln_params: [B, 6, d_model].
        """
        emb = sinusoidal_embedding_1d(t.float(), self.d_embed)
        time_emb = self.mlp(emb)
        adaln_flat = self.adaln_proj(time_emb)
        adaln_params = adaln_flat.view(-1, 6, self.d_model)
        return time_emb, adaln_params
