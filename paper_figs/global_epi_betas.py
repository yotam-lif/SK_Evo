import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from misc.cmn_sk import (
    init_h,
    init_J,
    relax_sk,
    compute_fit_slow,
    compute_dfe,
    compute_bdfe,
    compute_rank,
    compute_fit_off
)
from misc.cmn import curate_sigma_list
from misc.cmn import init_sigma
import scienceplots

# Use science plots for better plot aesthetics
plt.style.use('science')
num_betas = 3
betas = np.linspace(0.0, 1.0, num_betas, dtype=float)
colors = sns.color_palette('CMRmap', num_betas+1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue']
plt.rcParams['font.size'] = 14

# Create a larger figure with 5 subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=False)

def compute_metrics(N, random_state, beta, rho, num_points):
    sigma_0 = init_sigma(N)
    h = init_h(N, random_state=random_state, beta=beta)
    J = init_J(N, random_state=random_state, beta=beta, rho=rho)
    F_off = compute_fit_off(sigma_0, h, J)
    flip_seq = relax_sk(sigma_0, h, J)
    num_flips = len(flip_seq)
    ts = list(np.linspace(0, num_flips, num_points, dtype=int))
    sigmas = curate_sigma_list(sigma_0, flip_seq, ts)
    dfes = [compute_dfe(sigma, h, J) for sigma in sigmas]
    mean_dfes = [np.mean(dfe) for dfe in dfes]
    var_dfes = [np.var(dfe) for dfe in dfes]
    bdfes = [compute_bdfe(sigma, h, J)[0] for sigma in sigmas]
    mean_bdfes = [np.mean(bdfe) for bdfe in bdfes]
    var_bdfes = [np.var(bdfe) for bdfe in bdfes]
    fits = [compute_fit_slow(sigma, h, J, F_off) for sigma in sigmas]
    rank = [compute_rank(sigma, h, J) / N for sigma in sigmas]
    return fits, ts, mean_dfes, var_dfes, mean_bdfes, var_bdfes, rank

def main():
    N = 2000
    random_state = 42
    rho = 0.5
    num_points = 60
    num_repeats = 10

    # Create directory for saving plots
    output_dir = '../Plots/paper_figs'
    os.makedirs(output_dir, exist_ok=True)


    for beta, color in zip(betas, colors):
        fits, ts, mean_dfe, var_dfe, mean_bdfe, var_bdfe, rank = compute_metrics(N, random_state, beta, rho, num_points)
        max_fit = fits[-1]
        fits = [(fit / max_fit) * 100 for fit in fits]
        max_t = ts[-1]
        ts = [(t / max_t) * 100 for t in ts]

        # Plot mean_dfes
        sns.regplot(x=fits, y=mean_dfe, ax=axes[0, 0], marker='o', color=color, label=f'$\\beta={beta}$', scatter_kws={'s': 50})
        # axes[0, 0].plot(fits, mean_dfe, color=color, label=f'$\\beta={beta}$')
        axes[0, 0].set_xlabel('Fitness (\\% from maximum reached)')
        axes[0, 0].set_ylabel(r'$ \mathbb{E} \left [ P(\Delta) \right ] $')
        axes[0, 0].legend(fontsize=12, loc='upper right')

        # Plot var_dfes
        sns.regplot(x=fits, y=var_dfe, ax=axes[0, 1], marker='o', color=color, label=f'$\\beta={beta}$', scatter_kws={'s': 50})
        # axes[0, 1].plot(fits, var_dfe, color=color, label=f'$\\beta={beta}$')
        axes[0, 1].set_xlabel('Fitness (\\% from maximum reached)')
        axes[0, 1].set_ylabel(r'$ Var \left [ P(\Delta) \right ] $')
        axes[0, 1].legend(fontsize=12, loc='upper right')

        # Plot mean_bdfes
        sns.regplot(x=fits, y=mean_bdfe, ax=axes[0, 2], marker='o', color=color, label=f'$\\beta={beta}$', scatter_kws={'s': 50})
        # axes[0, 2].plot(fits, mean_bdfe, color=color, label=f'$\\beta={beta}$')
        axes[0, 2].set_xlabel('Fitness (\\% from maximum reached)')
        axes[0, 2].set_ylabel(r'$ \mathbb{E} \left [ P_+(\Delta) \right ] $')
        axes[0, 2].legend(fontsize=12, loc='upper right')

        # Plot var_bdfes
        sns.regplot(x=fits, y=var_bdfe, ax=axes[1, 0], marker='o', color=color, label=f'$\\beta={beta}$', scatter_kws={'s': 50})
        # axes[1, 0].plot(fits, var_bdfe, color=color, label=f'$\\beta={beta}$')
        axes[1, 0].set_xlabel('Fitness (\\% from maximum reached)')
        axes[1, 0].set_ylabel(r'$ Var \left [ P_+(\Delta) \right ] $')
        axes[1, 0].legend(fontsize=12, loc='upper right')

        # Plot ranks
        sns.regplot(x=fits, y=rank, ax=axes[1, 1], marker='o', color=color, label=f'$\\beta={beta}$', scatter_kws={'s': 50})
        axes[1, 1].set_xlabel('Fitness (\\% from maximum reached)')
        axes[1, 1].set_ylabel(r'$r(t)/ N$')
        axes[1, 1].legend(fontsize=12, loc='upper right')

    # Add legends
    for ax in axes.flat:
        ax.legend()
        ax.tick_params(axis='both', which='major', length=10, width=1, labelsize=14)
        ax.tick_params(axis='both', which='minor', length=5, width=1, labelsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    # Save the plot
    plot_path = os.path.join(output_dir, 'global_epi_betas.svg')
    plt.savefig(plot_path, format='svg', bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main()