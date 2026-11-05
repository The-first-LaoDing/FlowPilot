"""WanVAEEncoder: frozen Wan2.2 VAE + patch_embed Conv3D for FlowPilot."""

import torch
import torch.nn as nn
from flowpilot.models.vae2_2 import Wan2_2_VAE


class WanVAEEncoder(nn.Module):
    """Wraps frozen Wan2_2_VAE and adds a patch_embed Conv3D.

    encode()   : depth frames [B,N,1,H,W] → VAE latent [B,48,T',H',W']
    decode()   : VAE latent → depth frames [B,T,1,H,W] in [0,1]
    tokenize() : latent [B,48,T',H',W'] → tokens [B,T'*H'*W',512]

    patch_embed is the only non-frozen part; VAE weights stay frozen.
    """

    def __init__(
        self,
        vae_pth: str,
        z_dim: int = 48,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        super().__init__()
        self.z_dim = z_dim

        # Frozen VAE — hardcoded architecture matching Wan2.2-TI2V-5B.
        # Wan2_2_VAE is a plain Python class (not nn.Module); its internal
        # self.model is already frozen via .requires_grad_(False) in its __init__.
        # Store as a non-module attribute so nn.Module doesn't try to register it.
        vae = Wan2_2_VAE(
            vae_pth=vae_pth,
            z_dim=z_dim,
            c_dim=160,
            dim_mult=[1, 2, 4, 4],
            temperal_downsample=[False, True, True],
            dtype=dtype,
            device=device,
        )
        object.__setattr__(self, "vae", vae)  # bypass nn.Module.__setattr__

        # Trainable patch embedding: maps latent channels → video token dim
        self.patch_embed = nn.Conv3d(z_dim, 512, kernel_size=1, stride=1)

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode depth frames to VAE latent.

        Args:
            frames: [B, N, 1, H, W] depth in [0, 1].

        Returns:
            latent: [B, 48, T', H', W'] where T'=(N-1)//4+1, H'=H//16, W'=W//16.
        """
        B, N, _, H, W = frames.shape
        # Normalize [0,1] → [-1,1] and repeat to 3 channels
        x = frames * 2.0 - 1.0  # [B, N, 1, H, W]
        x = x.repeat(1, 1, 3, 1, 1)  # [B, N, 3, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, 3, N, H, W]

        # Wan2_2_VAE.encode() takes list of [3, T, H, W]
        inp = [x[b] for b in range(B)]  # list of [3, N, H, W]
        latents = self.vae.encode(inp)   # list of [48, T', H', W']
        if latents is None:
            raise RuntimeError("Wan2_2_VAE.encode() returned None — check input shape/dtype")
        return torch.stack(latents, dim=0)  # [B, 48, T', H', W']

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode VAE latent to depth frames.

        Args:
            latent: [B, 48, T', H', W'].

        Returns:
            frames: [B, T, 1, H, W] depth in [0, 1].
        """
        B = latent.shape[0]
        inp = [latent[b] for b in range(B)]  # list of [48, T', H', W']
        decoded = self.vae.decode(inp)        # list of [3, T, H, W] in [-1,1]
        if decoded is None:
            raise RuntimeError("Wan2_2_VAE.decode() returned None — check input shape/dtype")
        # Stack, take channel 0 (all identical), map [-1,1]→[0,1]
        out = torch.stack(decoded, dim=0)     # [B, 3, T, H, W]
        out = (out[:, 0:1] + 1.0) / 2.0      # [B, 1, T, H, W]
        out = out.clamp(0.0, 1.0)
        return out.permute(0, 2, 1, 3, 4)     # [B, T, 1, H, W]

    def tokenize(self, latent: torch.Tensor) -> torch.Tensor:
        """Apply patch_embed and flatten to sequence of tokens.

        Args:
            latent: [B, 48, T', H', W'] (noisy, post-teacher-forcing).

        Returns:
            tokens: [B, T'*H'*W', 512].
        """
        x = self.patch_embed(latent)          # [B, 512, T', H', W']
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)         # [B, T', H', W', 512]
        return x.reshape(B, T * H * W, C)     # [B, T'*H'*W', 512]
