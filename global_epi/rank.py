import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from misc.cmn_sk import (
    init_h,
    init_J,
    relax_sk,
    compute_fit_slow,
    compute_rank,
    compute_fit_off
)
from misc.cmn import curate_sigma_list
from misc.cmn import init_sigma
import scienceplots

# Use science plots for better plot aesthetics
plt.style.use('science')
betas = np.array([0.25, 0.5, 1.0], dtype=float)
num_betas = len(betas)
colors = sns.color_palette('CMRmap', num_betas+3)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue']

# Create a larger figure
plt.figure(figsize=(6, 4))

def approx_r_over_N(fit, N, beta, delta):
    return 0.5 * (1 - (fit * np.sqrt(np.pi / 2) / (N * delta * np.sqrt(1 - 1 * beta / 4))))

def main():
    # Parameters
    N = 2000
    random_state = 42
    rho = 1.0
    num_points = 50
    delta = 0.05

    # Create directory for saving plots
    output_dir = '../Plots/global_epi'
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_betas):
        # Initialize the model
        beta = betas[i]
        color = colors[i]
        sigma_0 = init_sigma(N)
        h = init_h(N, random_state=random_state, beta=beta, delta=delta)
        J = init_J(N, random_state=random_state, beta=beta, rho=rho, delta=delta)
        F_off = compute_fit_off(sigma_0, h, J)
        flip_seq = relax_sk(sigma_0, h, J)
        num_flips = len(flip_seq)
        ts = list(np.linspace(0, num_flips, num_points, dtype=int))
        sigmas = curate_sigma_list(sigma_0, flip_seq, ts)
        ranks = [compute_rank(sigma, h, J) / N for sigma in sigmas]
        fits = [compute_fit_slow(sigma, h, J, F_off) for sigma in sigmas]

        x = fits[int(len(fits)/2)]
        y = ranks[int(len(ranks)/2)]
        delta = 0.05

        # Plot mean_dfes as a function of fits
        sns.regplot(x=fits, y=ranks, marker='o', color=color, label=f'$\\beta={beta}$', scatter_kws={'s': 50})
        # slope, intercept, r_value, p_value, std_err = linregress(fits, mean_bdfes)
        # plt.text(x, y - delta - 0.07*i , f'$m \\times N = {N*slope:.2f}$', fontsize=14, color=color, ha='center', rotation=-17-8*i)

    ys_beta0 = [approx_r_over_N(x, N, 0, delta) for x in fits]
    ys_beta_half = [approx_r_over_N(x, N, 0.5, delta) for x in fits]
    ys_beta1 = [approx_r_over_N(x, N, 1, delta) for x in fits]
    sns.regplot(x=fits, y=ys_beta0, marker='o', color=colors[-3], label=f'approx; $\\beta={0}$', scatter_kws={'s': 50})
    sns.regplot(x=fits, y=ys_beta_half, marker='o', color=colors[-2], label=f'approx; $\\beta={0.5}$',
                scatter_kws={'s': 50})
    # sns.regplot(x=fits, y=ys_beta1, marker='o', color=colors[-1], label=f'approx; $\\beta={1}$', scatter_kws={'s': 50})
    plt.xlabel('Fitness', fontsize=16)
    plt.ylabel(r'$r(t)/N$', fontsize=16)
    plt.title(f'Rank as a function of fitness; N={N}', fontsize=16)
    plt.legend(fontsize=16, loc='lower left', frameon=True)
    plt.tight_layout()
    # Get the current axis and set tick parameters
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', length=10, width=1, labelsize=14)
    ax.tick_params(axis='both', which='minor', length=5, width=1, labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Save the plot
    plot_path = os.path.join(output_dir, 'ranks.png')
    plt.savefig(plot_path, format='png', bbox_inches='tight', dpi=500)
    plt.close()

if __name__ == "__main__":
    main()