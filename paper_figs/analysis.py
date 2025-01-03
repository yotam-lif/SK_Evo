import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import os
import scienceplots
from cmn.cmn_eqs import theta, rhs, normalize

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
output_dir = '../Plots/paper_figs'
os.makedirs(output_dir, exist_ok=True)
plt.style.use('science')

# Spatial grid
s = np.linspace(s_min, s_max, N_s)
ds = s[1] - s[0]
t_eval = np.linspace(t_min, t_max, T_num)

p0_gauss = np.exp(-(s ** 2) / (2 * sig ** 2))
p0_gauss /= np.sum(p0_gauss) * ds  # Normalize

p0_neg = p0_gauss * theta(-s)

s0 = 1.0  # Center of the Gaussian
p0_gauss_shifted = np.exp(-((s-s0) ** 2) / (2 * sig ** 2))
p0_gauss_shifted /= np.sum(p0_gauss_shifted) * ds  # Normalize

# Solve the PDE using solve_ivp with normalization after each step
solution_gaussian = solve_ivp(
    rhs, [t_min, t_max], p0_gauss, t_eval=t_eval, method='RK45',
    vectorized=False, events=None, args=(s, c, D)
)
p_all_g = np.apply_along_axis(normalize, 0, solution_gaussian.y)

solution_neg = solve_ivp(
    rhs, [t_min, t_max], p0_neg, t_eval=t_eval, method='RK45',
    vectorized=False, events=None, args=(s, c, D)
)
p_all_neg = np.apply_along_axis(normalize, 0, solution_neg.y)

solution_gaussian_shifted = solve_ivp(
    rhs, [t_min, t_max], p0_gauss_shifted, t_eval=t_eval, method='RK45',
    vectorized=False, events=None, args=(s, c, D)
)
p_all_g_shifted = np.apply_along_axis(normalize, 0, solution_gaussian_shifted.y)

# -----------------------------
# Plotting the solution at selected times
# -----------------------------
plt.figure(figsize=(10, 6))
time_indices = [0, len(t_eval) // 4, len(t_eval) // 2, 3 * len(t_eval) // 4, -1]
colors = ['blue', 'green', 'red']
labels = ['Gaussian', 'Negative Gaussian', 'Shifted Gaussian']

x_max = 40
x_min = -x_max
N_x = int((x_max - x_min) / ds)
x = np.linspace(x_min, x_max, N_x)

for idx in time_indices:
    if idx < p_all_g.shape[1]:  # Ensure the index is within bounds
        sol_g = p_all_g[:, idx]
        sol_neg = p_all_neg[:, idx]
        sol_g_shifted = p_all_g_shifted[:, idx]

        sol_g_on_x = np.interp(x, s, sol_g)
        sol_neg_on_x = np.interp(x, s, sol_neg)
        sol_g_shifted_on_x = np.interp(x, s, sol_g_shifted)

        plt.plot(x, sol_g_on_x, color=colors[0],
                 label=f'{labels[0]} t = {t_eval[idx]:.3f}' if idx == time_indices[0] else "")
        plt.plot(x, sol_neg_on_x, color=colors[1],
                 label=f'{labels[1]} t = {t_eval[idx]:.3f}' if idx == time_indices[0] else "")
        plt.plot(x, sol_g_shifted_on_x, color=colors[2],
                 label=f'{labels[2]} t = {t_eval[idx]:.3f}' if idx == time_indices[0] else "")

plt.title('Solution of the PDE at Different Times')
plt.xlabel('Position s')
plt.ylabel('Concentration p(s, t)')
plt.legend(fontsize=12, loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'analysis.svg'), format='svg', bbox_inches='tight')
plt.close()