import numpy as np

def theta(s):
    """Heaviside step function."""
    return 0.5 * (np.sign(s) + 1)


def negative_integral(s, p, ds, eps=1e-10):
    """Compute the integral of p(s) over negative part of s."""
    integrand = p
    # integrand *= _s
    integral = np.sum(integrand[s < 0]) * ds
    return integral if integral > eps else eps  # Prevent division by zero


def flip_term(s: np.ndarray, p: np.ndarray, ds: float) -> np.ndarray:
    dp_pos = theta(s) * p
    dp_neg = np.flip(dp_pos)
    flip_term = (dp_neg - dp_pos) * np.abs(s)
    flip_term /= negative_integral(s, p, ds)
    return flip_term


def drift_term(p, ds):
    dpdx = np.zeros_like(p)
    dpdx[1:] = (p[1:] - p[:-1]) / ds
    return dpdx


def diff_term(p, ds):
    dpdx2 = np.zeros_like(p)
    dpdx2[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / ds ** 2
    return dpdx2


def rhs(t, s, p, c, D):
    """Compute the RHS of the ODE system."""
    ds = s[1] - s[0]
    dpdt = np.zeros_like(p)
    dpdt -= c * drift_term(p, ds)
    dpdt += flip_term(s, p, ds)
    dpdt += D * diff_term(p, ds)
    # Apply boundary conditions (Dirichlet: p = 0 at boundaries)
    dpdt[0] = 0.0
    dpdt[-1] = 0.0
    return dpdt


def normalize(p, ds):
    """Ensure the solution remains normalized at each time step."""
    p /= np.sum(p) * ds  # Normalize the solution
    return p


def msd_fit_func(t, m, a):
    return m * t**a
