import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# Parameters
D = 0.2          # Diffusion coefficient
v = -2.0
s_max = 20.0     # Maximum s value
s_min = -s_max   # Minimum s value
N_s = 200        # Number of spatial grid points
t_min = 0.0      # Start time
t_max = 100.0     # End time

# Spatial grid
s = np.linspace(s_min, s_max, N_s)
ds = s[1] - s[0]

# Initial condition: Gaussian centered at s = s0
s0 = 0.0         # Center of the Gaussian
sigma = 1.0      # Standard deviation
p0 = np.exp(-((s - s0) ** 2) / (2 * sigma ** 2))
p0 /= (np.sqrt(2 * np.pi) * sigma)  # Normalize

# Boundary conditions: p = 0 at s_min and s_max (Dirichlet conditions)
p0[0] = 0.0
p0[-1] = 0.0

def theta(s):
    return 0.5 * (np.sign(s) + 1)

def nlt_term(s: np.ndarray, p: np.ndarray) -> np.ndarray:
    outflux = s * theta(s) * p
    influx = -s * theta(-s) * np.flip(p)
    return influx - outflux

# Function to compute the advection term using the upwind scheme
def advection_term(v, p, ds):
    dpdx = np.zeros_like(p)
    if v > 0:
        # Backward difference for v > 0
        dpdx[1:] = (p[1:] - p[:-1]) / ds
        dpdx[0] = 0.0  # Boundary condition at s_min
    elif v < 0:
        # Forward difference for v < 0
        dpdx[:-1] = (p[1:] - p[:-1]) / ds
        dpdx[-1] = 0.0  # Boundary condition at s_max
    # If v == 0, dpdx remains zero
    return -v * dpdx

# Function to compute the RHS of the ODE system
def rhs(t, p):
    # Second derivative approximation using finite differences
    dpdt = np.zeros_like(p)
    dpdt[1:-1] = D * (p[2:] - 2 * p[1:-1] + p[:-2]) / ds ** 2
    # Add drift term
    # dpdt += advection_term(v, p, ds)
    # Add transport term
    dpdt += nlt_term(s, p)
    # Apply boundary conditions (Dirichlet: p = 0 at boundaries)
    dpdt[0] = 0.0
    dpdt[-1] = 0.0
    return dpdt

# Time points where the solution is computed
t_eval = np.linspace(t_min, t_max, 200)

# Solve the PDE using solve_ivp
solution = solve_ivp(rhs, [t_min, t_max], p0, t_eval=t_eval, method='RK45')

# Extract the solution
p_all = solution.y  # Shape: (N_s, len(t_eval))

# -----------------------------
# Plotting the solution at selected times
# -----------------------------
plt.figure(figsize=(10, 6))
time_indices = [0, len(t_eval)//4, len(t_eval)//2, 3*len(t_eval)//4, -1]
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
    # Ensure normalization (optional)
    # p /= np.sum(p) * ds
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
    return m * t**a

# Exclude t=0 to avoid issues with log(0)
t_fit = t_eval[1:]
MSD_fit = MSD_change[1:]

# Perform the curve fitting
params, params_covariance = curve_fit(msd_fit_func, t_fit, MSD_fit, p0=[1.0, 1.0])

# Extract fitted parameters
m_fit, a_fit = params
print(f"Fitted parameters: m = {m_fit:.4f}, a = {a_fit:.4f}")

# Plot the fitted function
t_fit_line = np.linspace(t_min, t_max, 1000)
MSD_fit_line = msd_fit_func(t_fit_line, m_fit, a_fit)
plt.plot(t_fit_line, MSD_fit_line, '-', label=f'Fit: ΔMSD = {m_fit:.3f} * t^{a_fit:.3f}')
plt.legend()
plt.show()
