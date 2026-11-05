"""Polynomial trajectory utilities for Bernstein/power-basis math and
trajectory construction. Pure numpy; no rospy / no torch.

Provides:
  - Reference polynomial evaluation at arbitrary time (tracking, continuity).
  - Bernstein control points -> power basis coefficients.
  - Brake trajectory construction (goal-reached hover ramp-down).
"""

from math import comb as _comb

import numpy as np


def eval_poly(coefs_desc: np.ndarray, t: float, deriv: int = 0) -> np.ndarray:
    """Evaluate a polynomial (or its k-th derivative) at scalar time t.

    Coefficient layout is descending (highest power first):
        p(t) = coefs[0] * t^n + coefs[1] * t^(n-1) + ... + coefs[n]

    Args:
        coefs_desc: [N+1, D] float64.  N = polynomial degree, D = output dim (3 for xyz).
        t:          scalar time (seconds).
        deriv:      0 = position, 1 = velocity, 2 = acceleration.  Non-negative.

    Returns:
        [D] float64.  For deriv >= N+1 the polynomial is zero and a zero vector is returned.
    """
    n = coefs_desc.shape[0] - 1
    if deriv == 0:
        return np.array(sum(coefs_desc[n - k] * (t ** k) for k in range(n + 1)),
                        dtype=np.float64)
    out = np.zeros(coefs_desc.shape[1], dtype=np.float64)
    for k in range(n - deriv + 1):
        a_k_plus_d = coefs_desc[n - (k + deriv)]
        falling = 1
        for j in range(deriv):
            falling *= (k + deriv - j)
        out += a_k_plus_d * falling * (t ** k)
    return out


def bernstein_to_power_descending(b_pts: np.ndarray, T: float) -> np.ndarray:
    """Convert degree-n Bernstein control points parameterized on [0, T] to
    descending power-basis coefficients.

    Bernstein curve over tau in [0, 1], tau = t / T:
        p(tau) = sum_{i=0..n} b_pts[i] * B_{i,n}(tau)
    is rewritten as a power-basis polynomial in t:
        p(t) = coefs[0] * t^n + coefs[1] * t^(n-1) + ... + coefs[n]

    Args:
        b_pts: [N+1, D] float64 - Bernstein control points.
        T:     trajectory duration (seconds); must be > 0.

    Returns:
        coefs_desc: [N+1, D] float64 - descending power-basis coefficients
                    directly consumable by OMMPC polynomial_trajectory.h.
    """
    n = len(b_pts) - 1
    M = np.zeros((n + 1, n + 1), dtype=np.float64)
    for j in range(n + 1):
        for i in range(j + 1):
            M[j, i] = _comb(n, i) * _comb(n - i, j - i) * ((-1) ** (j - i))
    c_tau = M @ b_pts.astype(np.float64)                             # [N+1, D], ascending in tau
    a_t   = np.array([c_tau[j] / (T ** j) for j in range(n + 1)])     # [N+1, D], ascending in t
    return a_t[::-1].copy()                                           # descending


def build_brake_coefs(pos: np.ndarray, vel: np.ndarray, T_brake: float) -> np.ndarray:
    """Construct a degree-7 polynomial that decelerates (pos, vel) to rest at t=T_brake.

    Uses a quadratic ramp (only degrees 0, 1, 2 are nonzero):
      p(t)  = pos + vel * t - (vel / (2 * T_brake)) * t^2
      p(0)  = pos
      p'(0) = vel
      p'(T_brake) = 0   (velocity-matched stop at t=T_brake)

    Args:
        pos:     [3] world-frame starting position.
        vel:     [3] world-frame starting velocity.
        T_brake: seconds to decelerate to rest. Must be > 0.

    Returns:
        [8, 3] float64 descending power-basis coefficients
               (degrees 0..4 are zero; brake uses only degrees 0, 1, 2).
    """
    coefs = np.zeros((8, 3), dtype=np.float64)
    coefs[7] = pos
    coefs[6] = vel
    coefs[5] = -vel / (2.0 * T_brake)
    return coefs
