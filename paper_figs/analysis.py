import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import seaborn as sns
import os
import scienceplots
from cmn.uncmn_eqs import theta, flip_term, drift_term, diff_term
import pickle
from cmn import cmn, cmn_sk
from scipy.special import airy
import matplotlib.ticker as ticker
from scipy.stats import kstest, expon
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue']
plt.rcParams['font.size'] = 16
plt.style.use('science')

def create_subfig_a(ax):
    # Parameters
    s_max = 60.0  # Maximum s value
    s_min = -s_max  # Minimum s value
    N_s = 400  # Number of spatial grid points
    t_min = 0.0  # Start time
    t_max = 2.0  # End time
    c = 2.0  # Speed of the drift term
    D = 16.0
    sig = 2.0  # Standard deviation
    T_num = 200  # Number of time points
    num_points = 5

    # Spatial grid
    s = np.linspace(s_min, s_max, N_s)
    ds = s[1] - s[0]
    t_eval = np.linspace(t_min, t_max, T_num)

    # Initial condition: Gaussian
    p0_gauss = np.exp(-(s ** 2) / (2 * sig ** 2))
    p0_gauss /= np.sum(p0_gauss) * ds  # Normalize

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

    x_min = -25
    x_max = 10
    N_x = int((x_max - x_min) / ds)
    x = np.linspace(x_min, x_max, N_x)

    time_indices = np.linspace(0, len(t_eval) - 1, num_points, dtype=int)
    colors = sns.color_palette('CMRmap', n_colors=num_points)
    labels = [f't = {t_eval[idx]:.1f}' for idx in time_indices]

    for idx, color, label in zip(time_indices, colors, labels):
        if idx < p_all_g.shape[1]:  # Ensure the index is within bounds
            sol_g = p_all_g[:, idx]
            sol_g_on_x = np.interp(x, s, sol_g)
            ax.plot(x, sol_g_on_x, color=color, label=label, linewidth=3)

    ax.set_xlabel(r'$\Delta$', fontsize=14)
    ax.set_ylabel(r'$P(\Delta, t)$', fontsize=14)
    ax.legend(fontsize=14, loc='upper left', frameon=True)

def create_subfig_b(ax):
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
    colors = sns.color_palette('CMRmap', n_colors=4)
    # labels = [r'$P(\Delta, 0) \propto e^{-\frac{\Delta^2}{2\sigma^2}}$',
    #           r'$P(\Delta, 0) \propto e^{-\frac{\Delta^2}{2\sigma^2}} \cdot \theta(-\Delta)$',
    #           r'$P(\Delta, 0) \propto e^{-\frac{(\Delta - \Delta_0)^2}{2\sigma^2}}$']
    labels = ['Gaussian', 'Half Gaussian', 'Shifted Gaussian']
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
                    label=f'{labels[0]}' if idx == time_indices[0] else "", linewidth=3)
            ax.plot(x, sol_neg_on_x, color=colors[1], linestyle=linestyles[1],
                    label=f'{labels[1]}' if idx == time_indices[0] else "", linewidth=3)
            ax.plot(x, sol_g_shifted_on_x, color=colors[2], linestyle=linestyles[2],
                    label=f'{labels[2]}' if idx == time_indices[0] else "", linewidth=3)

    # ax.set_title(r'$P(\Delta, t)$ for different initial conditions')
    ax.set_xlabel(r'$\Delta$', fontsize=14)
    ax.set_ylabel(r'$P(\Delta, t)$', fontsize=14)
    ax.legend(fontsize=14, loc='upper left', frameon=True, title='Initial condition')
    # ax.grid(True)

def create_subfig_c(ax):
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
    sigmas = [1.0]
    p0_gaussians = [np.exp(-(s ** 2) / (2 * sigma ** 2)) for sigma in sigmas]
    delta_func = np.zeros_like(s)
    delta_func[len(s) // 2] = 1.0
    p0_gaussians.append(delta_func)

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
    colors = sns.color_palette('CMRmap', n_colors=3)
    labels = ['Gaussian', 'Dirac Delta']
    linestyles = ['-', '--']

    for idx in time_indices:
        for i, p_all_i in enumerate(p_all):
            if idx < p_all_i.shape[1]:  # Ensure the index is within bounds
                sol_on_x = np.interp(x, s, p_all_i[:, idx])
                ax.plot(x, sol_on_x, color=colors[i], linestyle=linestyles[i],
                        label=f'{labels[i]}' if idx == time_indices[0] else "", linewidth=3)

    # ax.set_title(r'Evolution of $P(\Delta, t)$ with different variances')
    ax.set_xlabel(r'$\Delta$')
    ax.set_ylabel(r'$P(\Delta, t)$')
    ax.legend(fontsize=14, loc='upper left', frameon=True, title='Initial condition')

def create_subfig_d(ax):
    from scipy.interpolate import interp1d

    # Load data
    file_path = '../misc/run_data/N4000_rho100_beta100_repeats50.pkl'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Extract and filter bdfes
    del_max = 0.8
    bdfes = []
    for repeat in data:
        h, J = repeat['h'], repeat['J']
        alpha = cmn.compute_sigma_from_hist(
            repeat['init_alpha'], repeat['flip_seq'],
            int(0.8 * len(repeat['flip_seq']))
        )
        vals = cmn_sk.compute_bdfe(alpha, h, J)[0]
        bdfes.extend(vals[vals < del_max])

    color = sns.color_palette('CMRmap', n_colors=5)
    # Plot histogram
    num_bins = 50
    sns.histplot(
        bdfes, ax=ax, bins=num_bins, stat='density', color=color[2],
        element="step", edgecolor='black', alpha=0.8, label='Simulation data'
    )

    def exp_pdf(x: np.ndarray, D, P0):
        return P0 * np.exp(-x / (P0 * D / 2))

    def neg_loglike_exp(params, data):
        D, P0 = params
        if D <= 0 or P0 <= 0:
            return np.inf
        x_grid = np.linspace(0, max(data) * 1.1, 2000)
        pdf_grid = exp_pdf(x_grid, D, P0)
        norm = cumulative_trapezoid(pdf_grid, x_grid, initial=0)[-1]
        if norm <= 0:
            return np.inf
        pdf_vals = exp_pdf(data, D, P0) / norm
        pdf_vals = np.maximum(pdf_vals, 1e-12)  # to avoid log(0)
        return -np.sum(np.log(pdf_vals))


    # Airy PDF (unnormalized)
    def airy_pdf(x: np.ndarray, D, P0):
        #  Ai(0), scaling c = P0/Ai(0), argument = x / sqrt(c*D)
        a0 = airy(0)[0]
        c = P0 / a0
        arg = x / np.sqrt(c * D/2)
        return np.where(x < 0, 0.0, c * airy(arg)[0])

    # Negative log-likelihood for Airy
    def neg_loglike_airy(params, data):
        D, P0 = params
        if D <= 0 or P0 <= 0:
            return np.inf
        x_grid = np.linspace(0, max(data) * 1.1, 2000)
        pdf_grid = airy_pdf(x_grid, D, P0)
        norm = cumulative_trapezoid(pdf_grid, x_grid, initial=0)[-1]
        if norm <= 0:
            return np.inf
        pdf_vals = airy_pdf(data, D, P0) / norm
        pdf_vals = np.maximum(pdf_vals, 1e-12)  # to avoid log(0)
        return -np.sum(np.log(pdf_vals))

    # MLE for Exp
    init_guess_exp = [20.0, 5.0]
    opt = minimize(
        neg_loglike_exp, init_guess_exp, args=(np.array(bdfes),),
        method='L-BFGS-B', bounds=[(1e-9, None), (1e-9, None)]
    )
    if opt.success:
        D_exp, P0_exp = opt.x
    else:
        D_exp, P0_exp = 1.0, 1.0

    # Normalize for plotting
    x_full = np.linspace(0, max(bdfes), 1000)
    exp_grid = exp_pdf(x_full, D_exp, P0_exp)
    norm_factor = cumulative_trapezoid(exp_grid, x_full, initial=0)[-1] or 1.0
    exp_plot = exp_grid / norm_factor
    ax.plot(x_full, exp_plot, color=color[0], linestyle='--', linewidth=3, label='Exp fit')

    # Build exp CDF for KS test
    x_cdf = np.linspace(0, max(bdfes), 1000)
    pdf_cdf = exp_pdf(x_cdf, D_exp, P0_exp)
    pdf_cdf /= cumulative_trapezoid(pdf_cdf, x_cdf, initial=0)[-1] or 1.0
    cdf_cumsum = cumulative_trapezoid(pdf_cdf, x_cdf, initial=0)
    exp_cdf = interp1d(x_cdf, cdf_cumsum, bounds_error=False, fill_value=(0.0, 1.0))

    # MLE for Airy
    init_guess_airy = [20.0, 5.0]
    opt = minimize(
        neg_loglike_airy, init_guess_airy, args=(np.array(bdfes),),
        method='L-BFGS-B', bounds=[(1e-9, None), (1e-9, None)]
    )
    if opt.success:
        D_airy, P0_airy = opt.x
    else:
        D_airy, P0_airy = 1.0, 1.0

    # Normalize for plotting
    x_full = np.linspace(0, max(bdfes), 1000)
    airy_grid = airy_pdf(x_full, D_airy, P0_airy)
    norm_factor = cumulative_trapezoid(airy_grid, x_full, initial=0)[-1] or 1.0
    airy_plot = airy_grid / norm_factor
    ax.plot(x_full, airy_plot, color=color[1], linestyle='--', linewidth=3, label='Airy fit')

    # Build Airy CDF for KS test
    x_cdf = np.linspace(0, max(bdfes), 1000)
    pdf_cdf = airy_pdf(x_cdf, D_airy, P0_airy)
    pdf_cdf /= cumulative_trapezoid(pdf_cdf, x_cdf, initial=0)[-1] or 1.0
    cdf_cumsum = cumulative_trapezoid(pdf_cdf, x_cdf, initial=0)
    airy_cdf = interp1d(x_cdf, cdf_cumsum, bounds_error=False, fill_value=(0.0, 1.0))

    # Perform K-S tests
    ks_exp = kstest(bdfes, exp_cdf)
    ks_airy = kstest(bdfes, airy_cdf)

    # Annotate plot
    ax.text(
        0.40, 0.66, rf'$D_{{exp}} = {D_exp:.2f}$',
        transform=ax.transAxes, color=color[0], va='top'
    )
    ax.text(
        0.40, 0.59, rf'$D_{{airy}} = {D_airy:.2f}$',
        transform=ax.transAxes, color=color[1], va='top'
    )
    ax.text(
        0.40, 0.52, rf'$p_{{exp}} = {ks_exp.pvalue:.2f}$',
        transform=ax.transAxes, color=color[0], va='top'
    )
    ax.text(
        0.40, 0.45, rf'$p_{{airy}} = {ks_airy.pvalue:.2f}$',
        transform=ax.transAxes, color=color[1], va='top'
    )

    # Final plot settings
    ax.set_xlabel(r'$\Delta$', fontsize=14)
    ax.set_ylabel(r'$P_+(\Delta,t)$', fontsize=14)
    ax.set_xlim(0, del_max)
    ax.legend(fontsize=14, loc='upper right', frameon=True)


def main():
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    # Create directory if it doesn't exist
    output_dir = '../Plots/paper_figs'
    os.makedirs(output_dir, exist_ok=True)

    create_subfig_a(axs[0, 0])
    create_subfig_b(axs[0, 1])
    create_subfig_c(axs[1, 0])
    create_subfig_d(axs[1, 1])

    labels = ['A', 'B', 'C', 'D']
    for i, ax in enumerate(axs.flat):
        ax.text(-0.1, 1.1, labels[i], transform=ax.transAxes,
                fontsize=20, fontweight='bold', va='top', ha='right')
        ax.tick_params(axis='both', which='major', length=10, width=1, labelsize=14)
        ax.tick_params(axis='both', which='minor', length=5, width=1, labelsize=14)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune='both', nbins=4))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune='both', nbins=4))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    plt.savefig(os.path.join(output_dir, 'analysis.svg'), format='svg', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()