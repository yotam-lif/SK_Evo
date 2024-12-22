import os
import numpy as np
import matplotlib.pyplot as plt
from misc.cmn_sk import (
    init_h,
    init_J,
    relax_sk,
    compute_fit_slow,
    calc_DFE,
)
from misc.cmn import curate_sigma_list
from misc.cmn import init_sigma

def main():
    # Parameters
    N = 2000
    random_state = 42
    beta = 1.0
    rho = 1.0
    num_points = 50

    # Create directory for saving plots
    output_dir = '../Plots/global_epi'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the model
    sigma_0 = init_sigma(N)
    h = init_h(N, random_state=random_state, beta=beta)
    J = init_J(N, random_state=random_state, beta=beta, rho=rho)

    flip_seq = relax_sk(sigma_0, h, J)
    num_flips = len(flip_seq)
    points = np.linspace(0, num_flips, num_points, dtype=int)
    sigmas = curate_sigma_list(sigma_0, flip_seq, points)
    dfes = [calc_DFE(sigma, h, J) for sigma in sigmas]
    mean_dfes = [np.mean(dfe) for dfe in dfes]
    fits = [compute_fit_slow(sigma, h, J) for sigma in sigmas]

    # Plot mean_dfes as a function of fits
    plt.figure(figsize=(10, 6))
    plt.plot(fits, mean_dfes, marker='o', linestyle='-', color='b')
    plt.xlabel('Fitness', fontsize=14)
    plt.ylabel('Mean DFE', fontsize=14)
    plt.title('Mean DFE as a function of Fitness', fontsize=16)
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(output_dir, 'mean_dfe.png')
    plt.savefig(plot_path, format='png', dpi=500)
    plt.close()

if __name__ == "__main__":
    main()