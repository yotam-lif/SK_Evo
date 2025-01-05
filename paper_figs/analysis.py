import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import seaborn as sns
import os
import scienceplots
from cmn.uncmn_eqs import theta, flip_term, drift_term, diff_term
import pickle
from cmn import cmn, cmn_sk
from scipy.special import airy
from scipy.optimize import curve_fit



plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue']
plt.rcParams['font.size'] = 14
plt.style.use('science')

def create_subfig_a(ax):
    # Parameters
    s_max = 60.0  # Maximum s value
    s_min = -s_max  # Minimum s value
    N_s = 400  # Number of spatial grid points
    t_min = 0.0  # Start time
    t_max = 6.0  # End time
    c = 1.0  # Speed of the drift term
    D = 4.0
    sig = 2.0  # Standard deviation
    T_num = 200  # Number of time points

    # Spatial grid
    s = np.linspace(s_min, s_max, N_s)
    ds = s[1] - s[0]
    t_eval = np.linspace(t_min, t_max, T_num)

    p0_gauss = np.exp(-(s ** 2) / (2 * sig ** 2))
    p0_gauss /= np.sum(p0_gauss) * ds  # Normalize

    p0_neg = p0_gauss * theta(-s)

    s0 = 3.0  # Center of the Gaussian
    p0_gauss_shifted = np.exp(-((s - s0) ** 2) / (2 * sig ** 2))
    p0_gauss_shifted /= np.sum(p0_gauss_shifted) * ds  # Normalize

    def rhs(t, p):
        """Compute the RHS of the ODE system."""
        dpdt = np.zeros_like(p)
        dpdt += drift_term(p, ds, c)
        dpdt += flip_term(s, p)
        dpdt += diff_term(p, ds, D)
        # Apply boundary conditions (Dirichlet: p = 0 at boundaries)
        dpdt[0] = 0.0
        dpdt[-1] = 0.0
        return dpdt

    def normalize(p):
        """Ensure the solution remains normalized at each time step."""
        p /= np.sum(p) * ds  # Normalize the solution
        return p

    # Solve the PDE using solve_ivp with normalization after each step
    solution_gaussian = solve_ivp(
        rhs, [t_min, t_max], p0_gauss, t_eval=t_eval, method='RK45',
        vectorized=False, events=None
    )
    p_all_g = np.apply_along_axis(normalize, 0, solution_gaussian.y)

    solution_neg = solve_ivp(
        rhs, [t_min, t_max], p0_neg, t_eval=t_eval, method='RK45',
        vectorized=False, events=None
    )
    p_all_neg = np.apply_along_axis(normalize, 0, solution_neg.y)

    solution_gaussian_shifted = solve_ivp(
        rhs, [t_min, t_max], p0_gauss_shifted, t_eval=t_eval, method='RK45',
        vectorized=False, events=None
    )
    p_all_g_shifted = np.apply_along_axis(normalize, 0, solution_gaussian_shifted.y)

    x_min = -30
    # x_max = -x_min
    x_max = 15
    N_x = int((x_max - x_min) / ds)
    x = np.linspace(x_min, x_max, N_x)

    time_indices = [0, len(t_eval) // 3, -1]
    colors = sns.color_palette('CMRmap', n_colors=5)
    labels = [r'$P(\Delta, 0) \propto e^{-\frac{\Delta^2}{2\sigma^2}}$',
              r'$P(\Delta, 0) \propto e^{-\frac{\Delta^2}{2\sigma^2}} \cdot \theta(-\Delta)$',
              r'$P(\Delta, 0) \propto e^{-\frac{(\Delta - \Delta_0)^2}{2\sigma^2}}$']
    linestyles = ['-', '--', ':']

    for idx in time_indices:
        if idx < p_all_g.shape[1]:  # Ensure the index is within bounds
            sol_g = p_all_g[:, idx]
            sol_neg = p_all_neg[:, idx]
            sol_g_shifted = p_all_g_shifted[:, idx]

            sol_g_on_x = np.interp(x, s, sol_g)
            sol_neg_on_x = np.interp(x, s, sol_neg)
            sol_g_shifted_on_x = np.interp(x, s, sol_g_shifted)

            ax.plot(x, sol_g_on_x, color=colors[0], linestyle=linestyles[0],
                    label=f'{labels[0]}' if idx == time_indices[0] else "", linewidth=1.5)
            ax.plot(x, sol_neg_on_x, color=colors[3], linestyle=linestyles[1],
                    label=f'{labels[1]}' if idx == time_indices[0] else "", linewidth=1.5)
            ax.plot(x, sol_g_shifted_on_x, color=colors[2], linestyle=linestyles[2],
                    label=f'{labels[2]}' if idx == time_indices[0] else "", linewidth=1.5)

    # ax.set_title(r'$P(\Delta, t)$ for different initial conditions')
    ax.set_xlabel(r'$\Delta$')
    ax.set_ylabel(r'$P(\Delta, t)$')
    ax.legend(fontsize=12, loc='upper left', frameon=True)
    # ax.grid(True)

def create_subfig_b(ax):
    # Parameters
    s_max = 30.0  # Maximum s value
    s_min = -s_max  # Minimum s value
    N_s = 600  # Number of spatial grid points
    t_min = 0.0  # Start time
    t_max = 3.0  # End time
    c = 1.0  # Speed of the drift term
    D = 4.0
    T_num = 200  # Number of time points

    # Spatial grid
    s = np.linspace(s_min, s_max, N_s)
    ds = s[1] - s[0]
    t_eval = np.linspace(t_min, t_max, T_num)

    # Initial conditions with different variances
    sigmas = [1.0, 0.1, 0.05]
    p0_gaussians = [np.exp(-(s ** 2) / (2 * sigma ** 2)) for sigma in sigmas]

    def rhs(t, p):
        """Compute the RHS of the ODE system."""
        dpdt = np.zeros_like(p)
        dpdt += drift_term(p, ds, c)
        dpdt += flip_term(s, p)
        dpdt += diff_term(p, ds, D)
        # Apply boundary conditions (Dirichlet: p = 0 at boundaries)
        dpdt[0] = 0.0
        dpdt[-1] = 0.0
        return dpdt

    def normalize(p):
        """Ensure the solution remains normalized at each time step."""
        p /= np.sum(p) * ds  # Normalize the solution
        return p

    # Solve the PDE using solve_ivp with normalization after each step
    solutions = [solve_ivp(rhs, [t_min, t_max], p0, t_eval=t_eval, method='RK45', vectorized=False, events=None) for p0
                 in p0_gaussians]
    p_all = [np.apply_along_axis(normalize, 0, sol.y) for sol in solutions]
    # p_all = [sol.y for sol in solutions]

    x_max = 5
    x_min = -20
    N_x = int((x_max - x_min) / ds)
    x = np.linspace(x_min, x_max, N_x)

    time_indices = [len(t_eval) // 2, -1]
    colors = sns.color_palette('CMRmap', n_colors=5)
    labels = [r'$\sigma = 10^{0}$',
              r'$\sigma = 10^{-1}$',
              r'$\sigma = 5 \times 10^{-2}$']
    linestyles = ['-', '--', ':']

    for idx in time_indices:
        for i, p_all_i in enumerate(p_all):
            if idx < p_all_i.shape[1]:  # Ensure the index is within bounds
                sol_on_x = np.interp(x, s, p_all_i[:, idx])
                ax.plot(x, sol_on_x, color=colors[i], linestyle=linestyles[i],
                        label=f'{labels[i]}' if idx == time_indices[0] else "", linewidth=1.5)

    # ax.set_title(r'Evolution of $P(\Delta, t)$ with different variances')
    ax.set_xlabel(r'$\Delta$')
    ax.set_ylabel(r'$P(\Delta, t)$')
    ax.legend(fontsize=12, loc='upper left', frameon=True)

def create_subfig_c(ax):
    N = 4000
    color = sns.color_palette('CMRmap', n_colors=3)
    file_path = '../misc/run_data/N4000_rho100_beta100_repeats50.pkl'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    del_max = 0.7
    bdfes = []
    num_bins = 30
    for repeat in data:
        h = repeat['h']
        J = repeat['J']
        flip_seq = repeat['flip_seq']
        sig_init = repeat['init_alpha']
        t = int(0.8 * len(flip_seq))
        alpha = cmn.compute_sigma_from_hist(sig_init, flip_seq, t)
        bdfe = cmn_sk.compute_bdfe(alpha, h, J)[0]
        bdfes.extend(bdfe[bdfe < del_max])

    sns.histplot(bdfes, ax=ax, kde=False, bins=num_bins, label='Simulation data', stat='density', color=color[2],
                 element="step", edgecolor='black', alpha=1.0)

    def sol_airy(x, D, P_0):
        ai_0 = airy(0)[0]
        c = P_0 / ai_0
        Ai = airy(x / np.sqrt(c * D))[0]
        return (P_0 / ai_0) * Ai

    def sol_exp(x, D, P_0):
        return P_0 * np.exp(-x / (P_0 * D))

    # Fit sol_exp and sol_airy to the bdfe
    hist, bin_edges = np.histogram(bdfes, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    P_0 = hist[0]
    # Fit sol_exp
    popt_exp, _ = curve_fit(lambda x, D: sol_exp(x, D, P_0), bin_centers, hist, bounds=(0, np.inf))
    D_exp = popt_exp[0]
    ax.plot(bin_centers, sol_exp(bin_centers, D_exp, P_0), color=color[0], label='Exponential fit', linestyle='--', linewidth=1.5)
    # Fit sol_airy
    popt_airy, _ = curve_fit(lambda x, D: sol_airy(x, D, P_0), bin_centers, hist, bounds=(0, np.inf))
    D_airy = popt_airy[0]
    ax.plot(bin_centers, sol_airy(bin_centers, D_airy, P_0), color=color[1], label=f'Airy fit', linestyle='--', linewidth=1.5)

    # Calculate chi-squared values
    chi_squared_exp = np.sum(((hist - sol_exp(bin_centers, D_exp, P_0)) ** 2) / sol_exp(bin_centers, D_exp, P_0))
    chi_squared_airy = np.sum(((hist - sol_airy(bin_centers, D_airy, P_0)) ** 2) / sol_airy(bin_centers, D_airy, P_0))

    # Add text annotations for D_exp, D_airy, and chi-squared values
    ax.text(0.40, 0.66, f'$D_{{exp}} \\times N = {int(D_exp * N)}$', transform=ax.transAxes,
            verticalalignment='top', color=color[0])
    ax.text(0.40, 0.59, f'$D_{{airy}} \\times N = {int(D_airy * N)}$', transform=ax.transAxes,
            verticalalignment='top', color=color[1])
    ax.text(0.40, 0.52, f'$\\chi^2_{{exp}} = {chi_squared_exp:.2f}$', transform=ax.transAxes,
            verticalalignment='top', color=color[0])
    ax.text(0.40, 0.45, f'$\\chi^2_{{airy}} = {chi_squared_airy:.2f}$', transform=ax.transAxes,
            verticalalignment='top', color=color[1])

    ax.set_xlabel(r'$\Delta$', fontsize=14)
    ax.set_ylabel(r'$P_+ (\Delta, t)$', fontsize=14)
    ax.set_xlim(0, None)
    ax.legend(fontsize=12, loc='upper right', frameon=True)

    # Set y-ticks excluding 0
    y_ticks = ax.get_yticks()
    y_ticks = y_ticks[y_ticks != 0]
    ax.set_yticks(y_ticks)


def main():
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # Create directory if it doesn't exist
    output_dir = '../Plots/paper_figs'
    os.makedirs(output_dir, exist_ok=True)

    create_subfig_a(axs[0])
    create_subfig_b(axs[1])
    create_subfig_c(axs[2])

    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', length=10, width=1, labelsize=14)
        ax.tick_params(axis='both', which='minor', length=5, width=1, labelsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    plt.savefig(os.path.join(output_dir, 'analysis.svg'), format='svg', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()