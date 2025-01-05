import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
from cmn.uncmn_eqs import rhs, normalize

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

# Time points where the solution is computed
t_eval = np.linspace(t_min, t_max, T_num)

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