import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# Parameters
D = 0.2          # Diffusion coefficient
s_max = 20.0     # Maximum s value
s_min = -s_max   # Minimum s value
N_s = 200        # Number of spatial grid points
t_min = 0.0      # Start time
t_max = 50.0     # End time

# Spatial grid
s = np.linspace(s_min, s_max, N_s)
ds = s[1] - s[0]

# Initial condition: Gaussian centered at s = s0
s0 = 0.0         # Center of the Gaussian
sigma = 1.0      # Standard deviation
p0 = np.exp(-((s - s0) ** 2) / (2 * sigma ** 2))
p0 /= np.sum(p0) * ds  # Normalize

# Boundary conditions: p = 0 at s_min and s_max (Dirichlet conditions)
p0[0] = 0.0
p0[-1] = 0.0

def theta(s):
    """Heaviside step function."""
    return 0.5 * (np.sign(s) + 1)

def nlt_term(s: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Nonlinear transport term."""
    outflux = s * theta(s) * p
    influx = -s * theta(-s) * np.flip(p)
    return influx - outflux

def rhs(t, p):
    """Compute the RHS of the ODE system."""
    dpdt = np.zeros_like(p)
    dpdt[1:-1] = D * (p[2:] - 2 * p[1:-1] + p[:-2]) / ds ** 2
    dpdt += nlt_term(s, p)  # Add transport term
    dpdt[0] = 0.0  # Boundary condition at s_min
    dpdt[-1] = 0.0  # Boundary condition at s_max
    return dpdt

# Time points where the solution is computed
t_eval = np.linspace(t_min, t_max, 200)

# Callback function to normalize the solution at each time step
def normalize(p):
    """Ensure the solution remains normalized at each time step."""
    p /= np.sum(p) * ds  # Normalize the solution
    return p

# Solve the PDE using solve_ivp with normalization after each step
solution = solve_ivp(
    rhs, [t_min, t_max], p0, t_eval=t_eval, method='RK45',
    vectorized=False, events=None
)
p_all = np.apply_along_axis(normalize, 0, solution.y)

# -----------------------------
# Plotting the solution at selected times
# -----------------------------
plt.figure(figsize=(10, 6))
time_indices = [0, len(t_eval) // 4, len(t_eval) // 2, 3 * len(t_eval) // 4, -1]
for idx in time_indices:
    plt.plot(s, p_all[:, idx], label=f't = {t_eval[idx]:.3f}')
plt.title('Solution of the PDE at Different Times')
plt.xlabel('Position s')
plt.ylabel('Concentration p(s, t)')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Compute Mean Square Displacement (MSD)
# -----------------------------
MSD = np.zeros(len(t_eval))
for i in range(len(t_eval)):
    p = p_all[:, i]
    MSD[i] = np.sum(s**2 * p) * ds

# Compute the change in MSD from time 0
MSD_change = MSD - MSD[0]

# Plotting the change in MSD over time
plt.figure(figsize=(10, 6))
plt.plot(t_eval, MSD_change, 'o', label='ΔMSD data')
plt.title('Change in Mean Square Displacement over Time')
plt.xlabel('Time t')
plt.ylabel('ΔMSD(t) = MSD(t) - MSD(0)')
plt.grid(True)

# -----------------------------
# Fit ΔMSD to f(t) = m * t^a
# -----------------------------
def msd_fit_func(t, m, a):
    """Fitting function for MSD change."""
    return m * t**a

# Exclude t=0 to avoid log(0) issues
t_fit = t_eval[1:]
MSD_fit = MSD_change[1:]

# Perform curve fitting
params_msd, params_covariance_msd = curve_fit(msd_fit_func, t_fit, MSD_fit, p0=[1.0, 1.0])

# Extract fitted parameters
m_fit_msd, a_fit_msd = params_msd
print(f"Fitted MSD parameters: m = {m_fit_msd:.4f}, a = {a_fit_msd:.4f}")

# Plot the fitted function
t_fit_line = np.linspace(t_min, t_max, 1000)
MSD_fit_line = msd_fit_func(t_fit_line, m_fit_msd, a_fit_msd)
plt.plot(t_fit_line, MSD_fit_line, '-', label=f'Fit: ΔMSD = {m_fit_msd:.3f} * t^{a_fit_msd:.3f}')
plt.legend()
plt.show()
