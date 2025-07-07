import numpy as np

def theta(s):
    """Heaviside step function."""
    return 0.5 * (np.sign(s) + 1)


def positive_integral(s, p, ds, eps=1e-10):
    """Compute the integral of p(s) over positive part of s."""
    integrand = p
    integral = np.sum(integrand[s > 0]) * ds
    return integral if integral > eps else eps  # Prevent division by zero


def flip_term(s: np.ndarray, p: np.ndarray) -> np.ndarray:
    ds = s[1] - s[0]
    dp_pos = theta(s) * p
    dp_neg = np.flip(dp_pos)
    flip_term = (dp_neg - dp_pos) * np.abs(s)
    flip_term /= positive_integral(s, p, ds)
    return flip_term


def drift_term(p: np.ndarray, ds, c):
    dpdx = np.zeros_like(p)
    if c > 0:
        # Backward difference for c > 0
        dpdx[1:] = (p[1:] - p[:-1]) / ds
    elif c < 0:
        # Forward difference for c < 0
        dpdx[:-1] = (p[1:] - p[:-1]) / ds
    else:
        dpdx[:] = 0.0
    return c * dpdx

def diff_term(p: np.ndarray, ds, D):
    dpdx2 = np.zeros_like(p)
    dpdx2[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / ds ** 2
    return (D/2) * dpdx2

def rhs(t, p, s, ds, c, D):
    return flip_term(s, p) + drift_term(p, ds, c) + diff_term(p, ds, D)

def msd_fit_func(t, m, a):
    return m * t**a

def normalize(p: np.ndarray, ds: float) -> np.ndarray:
    """
    Normalize a probability density p defined on a grid of spacing ds,
    so that sum(p)*ds == 1.
    """
    total = np.sum(p) * ds
    if total > 0:
        return p / total
    else:
        return p


