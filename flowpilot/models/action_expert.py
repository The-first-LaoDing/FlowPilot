"""Action Expert components: StateActionEncoder, ActionExpertBlock, ActionDecoder."""

import math
import torch
import torch.nn as nn

from flowpilot.models.embed import sinusoidal_embedding_1d


class StateActionEncoder(nn.Module):
    """Encodes state_goal + v_cmd + noisy free Bernstein coeffs + prev trajectory into action tokens.

    Token layout: [state_goal(0) | v_cmd(1) | free_coeffs(2:7) | prev(7) | registers(8:12)] = 12 tokens.
    """

    def __init__(
        self,
        state_dim: int = 25,
        action_dim: int = 3,
        d_model: int = 256,
        num_actions: int = 5,
        num_registers: int = 4,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_registers = num_registers
        total_tokens = 1 + 1 + num_actions + 1 + num_registers  # 12 (with default num_actions=5)

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.v_cmd_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.prev_encoder = nn.Sequential(
            nn.Linear(8 * 3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.prev_null_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.registers = nn.Parameter(torch.randn(1, num_registers, d_model) * 0.02)

        # Fixed sinusoidal positional embedding
        pos = torch.arange(total_tokens, dtype=torch.float32)
        pe = self._sinusoidal_pe(pos, d_model)  # [total_tokens, 256]
        self.register_buffer('pos_embed', pe.unsqueeze(0))  # [1, total_tokens, 256]

    @staticmethod
    def _sinusoidal_pe(positions: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        args = positions[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(
        self,
        state_goal: torch.Tensor,
        v_cmd: torch.Tensor,
        actions: torch.Tensor,
        prev_ctrl_pts: torch.Tensor,
        prev_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            state_goal:     [B, 25].
            v_cmd:          [B]  最大速度，单位 m/s
            actions:        [B, 5, 3] noisy free Bernstein coefficients.
            prev_ctrl_pts:  [B, 8, 3] prev trajectory ctrl pts in current body frame.
            prev_mask:      [B] bool. True = use prev_encoded; False = use null embed.

        Returns:
            action_tokens: [B, 12, 256].
        """
        B = state_goal.shape[0]
        state_token = self.state_encoder(state_goal).unsqueeze(1)  # [B, 1, 256]
        v_emb = sinusoidal_embedding_1d(v_cmd * 100, 256)  # [B, 256]
        v_token = self.v_cmd_encoder(v_emb).unsqueeze(1)  # [B, 1, 256]
        action_tokens = self.action_encoder(actions)  # [B, num_actions, 256]

        prev_flat = prev_ctrl_pts.reshape(B, 24)
        prev_encoded = self.prev_encoder(prev_flat).unsqueeze(1)  # [B, 1, 256]
        prev_null = self.prev_null_embed.expand(B, 1, -1)
        mask_b = prev_mask.view(B, 1, 1).to(prev_encoded.dtype)
        prev_token = mask_b * prev_encoded + (1.0 - mask_b) * prev_null  # [B, 1, 256]

        registers = self.registers.expand(B, -1, -1)  # [B, 4, 256]

        tokens = torch.cat(
            [state_token, v_token, action_tokens, prev_token, registers], dim=1
        )  # [B, total_tokens, 256]
        tokens = tokens + self.pos_embed
        return tokens


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class ActionExpertBlock(nn.Module):
    """Single Action Expert block within a MoT layer.

    Works in D_action=256 residual stream. Projects to 512D video head space
    for joint attention via wan_action_qkv einsum, then projects back via wan_action_o.
    """

    def __init__(
        self,
        d_action: int = 256,
        d_video_head: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 1024,
    ):
        super().__init__()
        self.d_action = d_action
        self.d_video_head = d_video_head
        self.num_heads = num_heads
        self.head_dim = d_video_head // num_heads  # 64

        self.modulation_a = nn.Parameter(torch.zeros(1, 6, d_action))

        self.norm1 = nn.LayerNorm(d_action)
        self.norm2 = nn.LayerNorm(d_action)

        # MoT projection: 256D → 8×64D=512D (QKV combined)
        self.wan_action_qkv = nn.Parameter(
            torch.randn(3, num_heads, d_action, self.head_dim) * 0.02
        )

        # RMSNorm for Q/K in 512D video head space
        self.wan_action_norm_q = RMSNorm(d_video_head)
        self.wan_action_norm_k = RMSNorm(d_video_head)

        # Output projection: 512D → 256D
        self.wan_action_o = nn.Linear(d_video_head, d_action)

        self.ffn = nn.Sequential(
            nn.Linear(d_action, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_action),
        )

    def _get_modulation(self, adaln_params: torch.Tensor):
        combined = adaln_params + self.modulation_a
        return combined.chunk(6, dim=1)

    def pre_attention(self, x: torch.Tensor, adaln_params: torch.Tensor):
        """Compute Q/K/V via einsum projection into video head space.

        Args:
            x: [B, S, 256] action tokens.
            adaln_params: [B, 6, 256].

        Returns:
            q, k, v: each [B, S, 8, 64] in video head space.
            gate_attn: [B, 1, 256].
        """
        shift_attn, scale_attn, gate_attn, _, _, _ = self._get_modulation(adaln_params)

        x_norm = self.norm1(x) * (1.0 + scale_attn) + shift_attn  # [B, S, 256]

        # Equivalent to: torch.einsum("btd,knde->kbtne", x_norm, self.wan_action_qkv)
        # Rewritten as matmul + reshape so ONNX constant-folds W into a single
        # [D, K*N*E] weight and TRT lowers it to a standard Gemm. The 4D einsum
        # form trips a TRT-10.3 mis-conversion that breaks coeff_velocity.
        K, N, D, E = self.wan_action_qkv.shape  # 3, num_heads, d_action, head_dim
        B, S = x_norm.shape[:2]
        W = self.wan_action_qkv.permute(2, 0, 1, 3).reshape(D, K * N * E)
        a_qkv = torch.matmul(x_norm, W).reshape(B, S, K, N, E)
        a_q, a_k, a_v = a_qkv[:, :, 0], a_qkv[:, :, 1], a_qkv[:, :, 2]

        # RMSNorm on Q/K in 512D space
        B, S = a_q.shape[:2]
        a_q = self.wan_action_norm_q(a_q.reshape(B, S, -1)).view(B, S, self.num_heads, self.head_dim)
        a_k = self.wan_action_norm_k(a_k.reshape(B, S, -1)).view(B, S, self.num_heads, self.head_dim)

        return a_q, a_k, a_v, gate_attn

    def post_attention(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
        gate_attn: torch.Tensor,
        adaln_params: torch.Tensor,
    ) -> torch.Tensor:
        _, _, _, shift_ffn, scale_ffn, gate_ffn = self._get_modulation(adaln_params)

        B, S = attn_out.shape[:2]
        attn_flat = attn_out.reshape(B, S, self.d_video_head)
        attn_proj = self.wan_action_o(attn_flat)  # [B, S, 256]

        x = x + attn_proj * gate_attn

        x_norm2 = self.norm2(x) * (1.0 + scale_ffn) + shift_ffn
        x = x + self.ffn(x_norm2) * gate_ffn

        return x


class ActionDecoder(nn.Module):
    """Decodes action tokens to free Bernstein coefficient velocity predictions.

    2-way AdaLN + Linear(256→3) + slice [2:7].
    """

    def __init__(self, d_model: int = 256, action_dim: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        self.linear = nn.Linear(d_model, action_dim)

    def forward(self, action_tokens: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_tokens: [B, 12, 256].
            time_emb: [B, 256] from ActionTimestepEmbedder.

        Returns:
            coeff_velocity: [B, 5, 3].
        """
        mod = self.adaln_modulation(time_emb)  # [B, 2*D]
        shift, scale = mod.chunk(2, dim=-1)
        x = self.norm(action_tokens) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)  # [B, 12, 3]
        return x[:, 2:7]  # [B, 5, 3]
