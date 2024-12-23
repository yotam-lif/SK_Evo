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
    calc_F_off
)
from misc.cmn import curate_sigma_list
from misc.cmn import init_sigma

def main():
    # Parameters
    N = 1000
    random_state = 42
    rho = 1.0
    num_points = 400
    betas = [0.0, 0.5, 1.0]
    colors = ['b', 'g', 'r']

    # Create directory for saving plots
    output_dir = '../Plots/global_epi'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for beta, color in zip(betas, colors):
        # Initialize the model
        sigma_0 = init_sigma(N)
        h = init_h(N, random_state=random_state, beta=beta)
        J = init_J(N, random_state=random_state, beta=beta, rho=rho)
        F_off = calc_F_off(sigma_0, h, J)
        flip_seq = relax_sk(sigma_0, h, J)
        num_flips = len(flip_seq)
        ts = list(np.linspace(0, num_flips, num_points, dtype=int))
        sigmas = curate_sigma_list(sigma_0, flip_seq, ts)
        dfes = [compute_dfe(sigma, h, J) for sigma in sigmas]
        mean_dfes = [np.mean(dfe) for dfe in dfes]
        fits = [compute_fit_slow(sigma, h, J, F_off) for sigma in sigmas]

        # Plot mean_dfes as a function of fits
        sns.regplot(x=fits, y=mean_dfes, marker='o', color=color, label=f'$\\beta={beta}$', scatter_kws={'s': 50})

    plt.xlabel('Fitness', fontsize=14)
    plt.ylabel('Mean DFE', fontsize=14)
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(output_dir, 'mean_dfe.png')
    plt.savefig(plot_path, format='png', dpi=500)
    plt.close()

if __name__ == "__main__":
    main()