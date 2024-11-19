import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# Parameters
s_max = 100.0  # Maximum s value
s_min = -s_max  # Minimum s value
N_s = 600  # Number of spatial grid points
t_min = 0.0  # Start time
t_max = 5.0  # End time
c = 1.0  # Speed of the drift term
D = 4.0
sig = 2.0  # Standard deviation
T_num = 200  # Number of time points
eps = 1e-5  # Small number to avoid division by zero

# Create directory if it doesn't exist
output_dir = '../Plots/solve_transport_rw'
os.makedirs(output_dir, exist_ok=True)

# Spatial grid
s = np.linspace(s_min, s_max, N_s)
ds = s[1] - s[0]

def theta(_s):
    """Heaviside step function."""
    return 0.5 * (np.sign(_s) + 1)

def negative_integral(_s, _p):
    """Compute the integral of p(s) over negative part of s."""
    integrand = _p
    integral = np.sum(integrand[_s < 0]) * ds
    return integral if integral > eps else eps  # Prevent division by zero

def flip_term(_s: np.ndarray, _p: np.ndarray) -> np.ndarray:
    dp_neg = theta(-_s) * _p
    dp_neg *= np.abs(_s)
    dp_pos = np.flip(dp_neg)
    flip_term = dp_pos - dp_neg
    flip_term /= negative_integral(_s, _p)
    return flip_term

# Function to compute the advection term using the upwind scheme
def drift_term(_c, _p, _s, _ds):
    dpdx = np.zeros_like(_p)
    if _c > 0:
        # Backward difference for c > 0
        dpdx[1:] = (_p[1:] - _p[:-1]) / _ds
    elif _c < 0:
        # Forward difference for c < 0
        dpdx[:-1] = (_p[1:] - _p[:-1]) / _ds
    else:
        dpdx[:] = 0.0
    return _c * dpdx

def diff_term(_sig, _p, _ds):
    dpdx2 = np.zeros_like(_p)
    dpdx2[1:-1] = (_p[2:] - 2 * _p[1:-1] + _p[:-2]) / _ds ** 2
    return D * dpdx2

# Function to compute the RHS of the ODE system
def rhs(t, p):
    """Compute the RHS of the ODE system."""
    dpdt = np.zeros_like(p)
    dpdt -= drift_term(c, p, s, ds)
    dpdt += flip_term(s, p)
    dpdt += diff_term(sig, p, ds)
    # Apply boundary conditions (Dirichlet: p = 0 at boundaries)
    dpdt[0] = 0.0
    dpdt[-1] = 0.0
    return dpdt

# Time points where the solution is computed
t_eval = np.linspace(t_min, t_max, T_num)

# Callback function to normalize the solution at each time step
def normalize(p):
    """Ensure the solution remains normalized at each time step."""
    p /= np.sum(p) * ds  # Normalize the solution
    return p

# Initial condition: Gaussian centered at s = s0
s0 = 0.0  # Center of the Gaussian
p0_gauss = np.exp(-((s - s0) ** 2) / (2 * sig ** 2))
p0_gauss /= np.sum(p0_gauss) * ds  # Normalize

# Solve the PDE using solve_ivp with normalization after each step
solution_gaussian = solve_ivp(
    rhs, [t_min, t_max], p0_gauss, t_eval=t_eval, method='RK45',
    vectorized=False, events=None
)
p_all_g = np.apply_along_axis(normalize, 0, solution_gaussian.y)

# Find the index of s closest to 0
s0_index = np.argmin(np.abs(s - s0))

# Extract the values of the function at s=0 over time
p_at_s0 = p_all_g[s0_index, :]

# Plotting the value of the function at s=0 as a function of time
plt.figure(figsize=(10, 6))
plt.plot(t_eval, p_at_s0, 'o-', label='p(s=0, t)')
plt.title('Value of the Function at s=0 over Time')
plt.xlabel('Time t')
plt.ylabel('Concentration p(s=0, t)')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_dir, 'value_at_s0_plot.png'))
plt.close()