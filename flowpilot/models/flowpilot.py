"""Top-level FlowPilot Phase-2 inference model."""

from typing import Tuple
import torch
import torch.nn as nn

from flowpilot.models.embed import VideoTimestepEmbedder, ActionTimestepEmbedder
from flowpilot.models.rope3d import get_rotary_pos_embed
from flowpilot.models.video_expert import VideoOutputHead
from flowpilot.models.action_expert import StateActionEncoder, ActionDecoder
from flowpilot.models.mot_layer import MoTLayer


class FlowPilotPhase2(nn.Module):
    """Phase 2: joint video + action flow model in VAE latent space.

    Input:  noisy_video_latent [B,48,3,6,10] (teacher-forced)
            state_goal [B,25], v_cmd [B], noisy_actions [B,48,12]
            t_video [B], t_action [B]
    Output: (video_velocity [B,48,2,6,10], action_velocity [B,48,12])
    """

    def __init__(
        self,
        num_layers: int = 12,
        d_video: int = 512,
        d_action: int = 256,
        num_heads: int = 8,
        ffn_dim_video: int = 2048,
        ffn_dim_action: int = 1024,
        z_dim: int = 48,
        num_actions: int = 48,
        latent_t: int = 3,
        grid_h: int = 6,
        grid_w: int = 10,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_video = d_video
        self.d_action = d_action
        self.latent_t = latent_t
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_video_tokens = latent_t * grid_h * grid_w  # 180

        self.patch_embed = nn.Conv3d(z_dim, d_video, kernel_size=1, stride=1)

        self.video_time_embedder = VideoTimestepEmbedder(d_video, d_embed=256)
        self.action_time_embedder = ActionTimestepEmbedder(d_action, d_embed=256)
        self.state_action_encoder = StateActionEncoder(25, 3, d_action, num_actions, 4)

        self.layers = nn.ModuleList([
            MoTLayer(d_video, d_action, num_heads, ffn_dim_video, ffn_dim_action)
            for _ in range(num_layers)
        ])

        self.video_output_head = VideoOutputHead(
            d_video, z_dim=z_dim, latent_t=latent_t, grid_h=grid_h, grid_w=grid_w,
        )
        self.action_decoder = ActionDecoder(d_action, 3)

        rope = get_rotary_pos_embed([latent_t, grid_h, grid_w], d_video // num_heads)
        self.register_buffer('rope_cos', rope[0])
        self.register_buffer('rope_sin', rope[1])

    def _tokenize(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(latent)
        B, C, T, H, W = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)

    def forward(
        self,
        noisy_video_latent: torch.Tensor,
        state_goal: torch.Tensor,
        v_cmd: torch.Tensor,
        noisy_free_coeffs: torch.Tensor,
        t_video: torch.Tensor,
        t_action: torch.Tensor,
        prev_ctrl_pts: torch.Tensor,
        prev_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            noisy_video_latent: [B, 48, 3, 6, 10] — teacher-forced.
            state_goal: [B, 25].
            v_cmd: [B].
            noisy_free_coeffs: [B, 5, 3] — noisy free Bernstein coefficients.
            t_video: [B].
            t_action: [B].
            prev_ctrl_pts: [B, 8, 3] previous trajectory ctrl pts in current body frame.
                                      Zeros when prev_mask is False (ignored by encoder).
            prev_mask:     [B] bool  True = condition on prev; False = use null embed.

        Returns:
            (video_velocity [B,48,2,6,10], coeff_velocity [B,5,3]).
        """
        video_tokens  = self._tokenize(noisy_video_latent)          # [B, 180, 512]
        action_tokens = self.state_action_encoder(                   # [B, 12, 256]
            state_goal, v_cmd, noisy_free_coeffs, prev_ctrl_pts, prev_mask,
        )

        time_emb_v, adaln_v = self.video_time_embedder(t_video)
        time_emb_a, adaln_a = self.action_time_embedder(t_action)

        rope_freqs = (self.rope_cos, self.rope_sin)
        for layer in self.layers:
            video_tokens, action_tokens = layer(
                video_tokens, action_tokens, adaln_v, adaln_a, rope_freqs,
                num_video_tokens=self.num_video_tokens,  # 180
            )

        video_velocity = self.video_output_head(video_tokens, time_emb_v)
        coeff_velocity = self.action_decoder(action_tokens, time_emb_a)
        return video_velocity, coeff_velocity
