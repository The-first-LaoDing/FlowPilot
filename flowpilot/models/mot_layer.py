"""Mixture-of-Transformers layer with joint video+action attention."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from flowpilot.models.video_expert import VideoTransformerBlock
from flowpilot.models.action_expert import ActionExpertBlock
from flowpilot.models.rope3d import apply_rotary_emb


class MoTLayer(nn.Module):
    """Single MoT Transformer layer.

    Orchestrates joint attention between video and action streams:
    1. Video/Action pre-attention (AdaLN + QKV projection)
    2. RoPE on video Q/K; RMSNorm on action Q/K (no RoPE)
    3. Concatenate Q/K/V → joint Flash Attention
    4. Split output → video/action post-attention (residual + FFN)
    """

    def __init__(
        self,
        d_video: int = 512,
        d_action: int = 256,
        num_heads: int = 8,
        ffn_dim_video: int = 2048,
        ffn_dim_action: int = 1024,
    ):
        super().__init__()
        self.video_block = VideoTransformerBlock(d_video, num_heads, ffn_dim_video)
        self.action_block = ActionExpertBlock(d_action, d_video, num_heads, ffn_dim_action)

    def forward(
        self,
        video_tokens: torch.Tensor,
        action_tokens: Optional[torch.Tensor],
        adaln_params_v: torch.Tensor,
        adaln_params_a: Optional[torch.Tensor],
        rope_freqs: Tuple[torch.Tensor, torch.Tensor],
        num_video_tokens: int = 180,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            video_tokens: [B, S_v, 512]. S_v=180 for Phase 2.
            action_tokens: [B, 35, 256].
            adaln_params_v: [B, 6, 512].
            adaln_params_a: [B, 6, 256] or None.
            rope_freqs: (cos, sin) each [180, 64] from get_rotary_pos_embed.
            num_video_tokens: Number of video tokens that get RoPE (default 180).

        Returns:
            (video_out, action_out): Updated tokens, same shapes as input.
        """
        B = video_tokens.shape[0]
        S_v = video_tokens.shape[1]

        # ① Video pre-attention
        v_q, v_k, v_v, v_gate = self.video_block.pre_attention(video_tokens, adaln_params_v)

        # Apply RoPE to video tokens only (first num_video_tokens)
        v_q_rope = v_q[:, :num_video_tokens]
        v_k_rope = v_k[:, :num_video_tokens]
        v_q_rope, v_k_rope = apply_rotary_emb(v_q_rope, v_k_rope, rope_freqs)
        v_q = torch.cat([v_q_rope, v_q[:, num_video_tokens:]], dim=1) if S_v > num_video_tokens else v_q_rope
        v_k = torch.cat([v_k_rope, v_k[:, num_video_tokens:]], dim=1) if S_v > num_video_tokens else v_k_rope

        if action_tokens is not None:
            # ② Action pre-attention
            a_q, a_k, a_v, a_gate = self.action_block.pre_attention(action_tokens, adaln_params_a)

            # ③ Concatenate for joint attention
            all_q = torch.cat([v_q, a_q], dim=1)  # [B, S_v+35, 8, 64]
            all_k = torch.cat([v_k, a_k], dim=1)
            all_v = torch.cat([v_v, a_v], dim=1)
        else:
            all_q, all_k, all_v = v_q, v_k, v_v

        # ④ Scaled dot-product attention
        all_q = all_q.transpose(1, 2)  # [B, 8, S, 64]
        all_k = all_k.transpose(1, 2)
        all_v = all_v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(all_q, all_k, all_v)
        attn_out = attn_out.transpose(1, 2)  # [B, S, 8, 64]

        # ⑤ Split output
        v_attn = attn_out[:, :S_v]  # [B, S_v, 8, 64]

        # ⑥ Video post-attention
        video_out = self.video_block.post_attention(
            video_tokens, v_attn, v_gate, adaln_params_v
        )

        if action_tokens is not None:
            a_attn = attn_out[:, S_v:]  # [B, 35, 8, 64]
            action_out = self.action_block.post_attention(
                action_tokens, a_attn, a_gate, adaln_params_a
            )
        else:
            action_out = None

        return video_out, action_out
