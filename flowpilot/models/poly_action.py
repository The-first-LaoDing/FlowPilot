"""Polynomial action representation via degree-7 Bernstein basis."""

import numpy as np
import torch


def _bernstein_basis(u: np.ndarray, n: int) -> np.ndarray:
    """Bernstein basis matrix [M, n+1] for parameter values u in [0,1]."""
    from math import comb
    M, K = len(u), n + 1
    B = np.zeros((M, K))
    for k in range(K):
        B[:, k] = comb(n, k) * (u ** k) * ((1 - u) ** (n - k))
    return B


def _bernstein_deriv(u: np.ndarray, n: int, order: int, T: float) -> np.ndarray:
    """d^order/dt^order Bernstein basis matrix via degree-reduction recurrence.

    dB_{k,n}/dt = (n/T) * (B_{k-1,n-1} - B_{k,n-1})   [first derivative]
    Higher orders apply recurrence on lower-degree basis.
    """
    from math import comb
    M = len(u)
    K = n + 1

    # Compute basis at degree n-order (for the recurrence terminal)
    B_base = _bernstein_basis(u, n - order)  # [M, n-order+1]

    # Prefactor: n! / (n-order)! / T^order
    prefactor = 1.0
    for i in range(order):
        prefactor *= (n - i)
    prefactor /= T ** order

    # General r-th derivative: d^r B_{k,n}/du^r = prefactor * sum_{j=0}^r (-1)^{r-j} C(r,j) B_{k-j, n-r}
    dB = np.zeros((M, K))
    signs = np.array([(-1) ** (order - j) * comb(order, j) for j in range(order + 1)])
    for k in range(K):
        for j in range(order + 1):
            idx = k - j  # index into B_base columns (degree n-order)
            if 0 <= idx <= n - order:
                dB[:, k] += signs[j] * B_base[:, idx]
    return prefactor * dB


class BernsteinActionBasis:
    """Precomputed Bernstein polynomial basis for 3D trajectory representation.

    Degree-7 polynomial over T=1.6s horizon, sampled at 30Hz (48 waypoints).
    Control points: b_0..b_7, where b_0/b_1/b_2 are hard-constrained from
    initial conditions; b_3..b_7 are the 5 free predicted parameters.

    Args:
        degree: Polynomial degree (default 7, K=8 control points).
        T: Time horizon in seconds (default 1.6).
        num_waypoints: Number of output waypoints (default 48).
        hz: Waypoint frequency in Hz (default 30).
        device: Torch device for tensor storage.
    """

    def __init__(
        self,
        degree: int = 7,
        T: float = 1.6,
        num_waypoints: int = 48,
        hz: float = 30,
        device: str = "cpu",
    ):
        self.degree = degree
        self.T = T
        self.num_waypoints = num_waypoints
        self.K = degree + 1  # 8

        # Sample points u_i = (i/hz) / T
        t = np.arange(num_waypoints) / hz  # [48]
        u = t / T                           # [48] in [0, 1)

        # Position and derivative basis matrices [48, 8]
        V    = _bernstein_basis(u, degree)
        dV   = _bernstein_deriv(u, degree, order=1, T=T)
        ddV  = _bernstein_deriv(u, degree, order=2, T=T)
        dddV = _bernstein_deriv(u, degree, order=3, T=T)

        dev = torch.device(device)
        self.V           = torch.from_numpy(V).float().to(dev)
        self.dV          = torch.from_numpy(dV).float().to(dev)
        self.ddV         = torch.from_numpy(ddV).float().to(dev)
        self.dddV        = torch.from_numpy(dddV).float().to(dev)

    def to(self, device):
        """Move all buffers to device, return self."""
        self.V           = self.V.to(device)
        self.dV          = self.dV.to(device)
        self.ddV         = self.ddV.to(device)
        self.dddV        = self.dddV.to(device)
        return self

    def constrained_coeffs(self, init_pva: torch.Tensor) -> torch.Tensor:
        """Compute constrained control points b_0, b_1, b_2 from initial conditions.

        Args:
            init_pva: [B, 9] — p0 [0:3], v0 [3:6], a0 [6:9] (unnormalized, body frame).
                      p0 is always [0,0,0] in body frame; included for interface uniformity.

        Returns:
            constrained: [B, 3, 3] — rows are b_0, b_1, b_2; cols are x,y,z.
        """
        p0 = init_pva[:, 0:3]  # [B, 3]
        v0 = init_pva[:, 3:6]  # [B, 3]
        a0 = init_pva[:, 6:9]  # [B, 3]
        b0 = p0
        b1 = b0 + v0 * (self.T / 7.0)
        b2 = 2.0 * b1 - b0 + a0 * (self.T ** 2 / 42.0)
        return torch.stack([b0, b1, b2], dim=1)  # [B, 3, 3]

    def expand_to_12d(
        self, free: torch.Tensor, constrained: torch.Tensor
    ) -> torch.Tensor:
        """Expand free + constrained control points to 12D waypoints.

        Args:
            free: [B, 5, 3] free control points (b_3..b_7), unnormalized.
            constrained: [B, 3, 3] constrained control points (b_0, b_1, b_2).

        Returns:
            waypoints_12d: [B, 48, 12] — pos(3) + vel(3) + acc(3) + jerk(3).
        """
        all_coeffs = torch.cat([constrained, free], dim=1)  # [B, 8, 3]
        pos  = torch.einsum("tk,bkd->btd", self.V,    all_coeffs)
        vel  = torch.einsum("tk,bkd->btd", self.dV,   all_coeffs)
        acc  = torch.einsum("tk,bkd->btd", self.ddV,  all_coeffs)
        jerk = torch.einsum("tk,bkd->btd", self.dddV, all_coeffs)
        return torch.cat([pos, vel, acc, jerk], dim=-1)  # [B, 48, 12]
