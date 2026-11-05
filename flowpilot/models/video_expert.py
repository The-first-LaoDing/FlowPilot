"""Video Expert components: VideoTransformerBlock and VideoOutputHead."""

import torch
import torch.nn as nn


class VideoTransformerBlock(nn.Module):
    """Single Video Expert block within a MoT layer.

    Handles AdaLN modulation, Q/K/V projection, FFN.
    Joint attention is computed externally in MoTLayer.

    AdaLN modulation combines:
      - Global adaln_params_v from VideoTimestepEmbedder [B, 6, D]
      - Per-layer modulation_v parameter [1, 6, D]
    The 6 modulation values: shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, ffn_dim: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # AdaLN per-layer learnable modulation
        self.modulation_v = nn.Parameter(torch.zeros(1, 6, d_model))

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # QKV projection (standard linear, 512 → 3×512)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        # Output projection (after attention, 512 → 512)
        self.out_proj = nn.Linear(d_model, d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def _get_modulation(self, adaln_params: torch.Tensor):
        """Combine global AdaLN params with per-layer modulation.

        Args:
            adaln_params: [B, 6, D] from VideoTimestepEmbedder.
        Returns:
            6 tensors, each [B, 1, D] for broadcasting over sequence.
        """
        # [B, 6, D] + [1, 6, D] → [B, 6, D]
        combined = adaln_params + self.modulation_v
        # Split into 6 modulation values, each [B, 1, D]
        return combined.chunk(6, dim=1)

    def pre_attention(self, x: torch.Tensor, adaln_params: torch.Tensor):
        """Compute Q/K/V with AdaLN modulation before joint attention.

        Args:
            x: [B, S, D] video tokens.
            adaln_params: [B, 6, D] from timestep embedder.

        Returns:
            q: [B, S, N, head_dim]
            k: [B, S, N, head_dim]
            v: [B, S, N, head_dim]
            gate_attn: [B, 1, D] gating value for post-attention residual.
        """
        shift_attn, scale_attn, gate_attn, _, _, _ = self._get_modulation(adaln_params)

        # AdaLN normalize
        x_norm = self.norm1(x) * (1.0 + scale_attn) + shift_attn  # [B, S, D]

        # QKV
        B, S, D = x_norm.shape
        qkv = self.qkv_proj(x_norm).view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each [B, S, N, head_dim]

        return q, k, v, gate_attn

    def post_attention(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
        gate_attn: torch.Tensor,
        adaln_params: torch.Tensor,
    ) -> torch.Tensor:
        """Apply gated residual + FFN after joint attention.

        Args:
            x: [B, S, D] original video tokens (for residual).
            attn_out: [B, S, N, head_dim] attention output (video portion).
            gate_attn: [B, 1, D] gating from pre_attention.
            adaln_params: [B, 6, D].

        Returns:
            [B, S, D] updated video tokens.
        """
        _, _, _, shift_ffn, scale_ffn, gate_ffn = self._get_modulation(adaln_params)

        B, S = attn_out.shape[:2]
        # Output projection
        attn_flat = attn_out.reshape(B, S, self.d_model)  # [B, S, D]
        attn_proj = self.out_proj(attn_flat)

        # Gated residual
        x = x + attn_proj * gate_attn  # [B, S, D]

        # FFN with AdaLN
        x_norm2 = self.norm2(x) * (1.0 + scale_ffn) + shift_ffn
        x = x + self.ffn(x_norm2) * gate_ffn

        return x


class VideoOutputHead(nn.Module):
    """Converts video tokens to velocity field in VAE latent space.

    Linear(512→48) + reshape → [B, 48, T', H', W'] → slice [:,:,1:] → [B, 48, T'_pred, H', W'].
    T'_pred = T' - 1 (skip condition frame at index 0).
    """

    def __init__(
        self,
        d_model: int = 512,
        z_dim: int = 48,
        latent_t: int = 3,
        grid_h: int = 6,
        grid_w: int = 10,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.latent_t = latent_t
        self.grid_h = grid_h
        self.grid_w = grid_w

        self.norm = nn.LayerNorm(d_model)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        self.linear = nn.Linear(d_model, z_dim)

    def forward(self, video_tokens: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_tokens: [B, T'*H'*W', 512].
            time_emb: [B, 512].

        Returns:
            video_velocity: [B, 48, T'-1, H', W']  (prediction frames only).
        """
        B = video_tokens.shape[0]

        mod = self.adaln_modulation(time_emb)       # [B, 2*D]
        shift, scale = mod.chunk(2, dim=-1)
        x = self.norm(video_tokens) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        x = self.linear(x)                           # [B, T'*H'*W', 48]
        # Reshape: sequence → spatial grid, then permute channels to dim 1
        x = x.reshape(B, self.latent_t, self.grid_h, self.grid_w, self.z_dim)
        # [B, T', H', W', 48] → [B, 48, T', H', W']
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, 48, T', H', W']

        return x[:, :, 1:]                           # [B, 48, T'-1, H', W']
