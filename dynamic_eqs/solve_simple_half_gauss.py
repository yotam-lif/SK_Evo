import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
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

x_max = 40
x_min = -x_max
N_x = int((x_max - x_min) / ds)
x = np.linspace(x_min, x_max, N_x)

def theta(_s):
    """Heaviside step function."""
    return 0.5 * (np.sign(_s) + 1)

def negative_integral(_s, _p):
    """Compute the integral of p(s) over negative part of s."""
    integrand = _p
    # integrand *= _s
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
p0_neg = p0_gauss * theta(-s)

# Solve the PDE using solve_ivp with normalization after each step
solution_neg = solve_ivp(
    rhs, [t_min, t_max], p0_neg, t_eval=t_eval, method='RK45',
    vectorized=False, events=None
)
p_all_n = np.apply_along_axis(normalize, 0, solution_neg.y)

# -----------------------------
# Plotting the solution at selected times
# -----------------------------


plt.figure(figsize=(10, 6))
time_indices = [0, len(t_eval) // 4, len(t_eval) // 2, 3 * len(t_eval) // 4, -1]
for idx in time_indices:
    if idx < p_all_n.shape[1]:  # Ensure the index is within bounds
        sol = p_all_n[:, idx]
        sol_on_x = np.interp(x, s, sol)
        plt.plot(x, sol_on_x, label=f't = {t_eval[idx]:.3f}')
plt.title('Solution of the PDE at Different Times')
plt.xlabel('Position s')
plt.ylabel('Concentration p(s, t)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'solution_plot_n.png'))
plt.close()
