"""Flow Matching scheduler with shift parameter (linear interpolation formulation)."""

import torch


class FlowMatchScheduler:
    """Linear interpolation Flow Matching scheduler.

    Shift formula: σ_shifted = shift * σ / (1 + (shift-1) * σ)
    Forward: z_t = (1 - σ) * z_0 + σ * ε
    Velocity target: v* = ε - z_0
    """

    def __init__(
        self,
        shift: float = 5.0,
        num_train_timesteps: int = 1000,
    ):
        self.shift = shift
        self.num_train_timesteps = int(num_train_timesteps)

        # Build sigmas table: t_id ∈ [0, 999] → shifted sigma
        sigma_linear = torch.linspace(0.0, 1.0, self.num_train_timesteps)
        self.sigmas = self._shift_sigma(sigma_linear)

    def _shift_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """Apply shift transform: σ_shifted = shift * σ / (1 + (shift-1) * σ)."""
        return self.shift * sigma / (1.0 + (self.shift - 1.0) * sigma)

    def inference_timesteps(self, num_steps: int = 50) -> torch.Tensor:
        """Generate inference timestep schedule.

        Returns:
            timesteps: [num_steps+1] tensor, linspace(1.0, 0.0, num_steps+1).
        """
        return torch.linspace(1.0, 0.0, num_steps + 1)

    def inference_sigmas(self, num_steps: int = 50) -> torch.Tensor:
        """Generate shifted sigma schedule for inference Euler integration.

        Returns:
            sigmas: [num_steps+1] tensor of shifted sigma values, from σ=1.0 to σ=0.0.
        """
        t_linear = torch.linspace(1.0, 0.0, num_steps + 1)
        return self._shift_sigma(t_linear)
