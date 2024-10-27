import os
import numpy as np
import matplotlib.pyplot as plt

import Funcs
from Funcs import (
    init_alpha,
    init_h,
    init_J,
    relax_SK,
    compute_fit_slow
)

def main():
    # Parameters
    N = 1000  # Number of spins
    random_state = 42  # Seed for reproducibility
    beta = 1.0
    rho = 1.0  # Sparsity of the coupling matrix
    # Define the number of lowest ranks to plot
    n_times = 10  # Example value, adjust as needed
    n_bins = 30
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dir_path = os.path.join(dir_path, "Plots", "pair_corr_dist")
    os.makedirs(dir_path, exist_ok=True)

    # Initialize the model
    alpha = init_alpha(N)
    h = init_h(N, random_state=random_state, beta=beta)
    J = init_J(N, random_state=random_state, beta=beta, rho=rho)
    flips = np.linspace(0, int(N * 0.6), n_times, dtype=int)

    final_alpha, saved_alphas, saved_flips, saved_ranks = relax_SK(
        alpha=alpha.copy(),
        his=h,
        Jijs=J,
        flips=flips,
        sswm=True  # Change to False to use Glauber flip
    )

    pair_corr_list = np.zeros((n_times, N))
    for n_time in range(n_times):
        alpha_i = saved_alphas[n_time]
        # The pairwise terms sum per i can be rewritten as (lf_i)^2 - N*sig_J^2 = (lf_i)^2 - beta
        # Compute J[i, :] * alpha_i for all i
        lfs = Funcs.calc_basic_lfs(alpha_i, h, J)
        pair_corr = lfs**2  # Shape: (N,)
        pair_corr_list[n_time] = pair_corr

    mean_list = [time.mean() for time in pair_corr_list]
    fitness_list = [compute_fit_slow(alpha_i, h, J) for alpha_i in saved_alphas]

    # Plot the mean pair correlation
    plt.figure(figsize=(8, 6))
    plt.plot(fitness_list, mean_list, label="Mean Pair Correlation")
    plt.xlabel("Fitness")
    plt.ylabel("Mean Pair Correlation")
    plt.title("Mean Pair Correlation vs Fitness")
    plt.show()

    # Now, for n_times, plot the distribution of pair correlations as histogram
    # for n_time in range(n_times):
    #     plt.figure(figsize=(8, 6))
    #     mean_pair_corr = pair_corr_list[n_time].mean()
    #     plt.hist(pair_corr_list[n_time], density=True, bins=n_bins, alpha=0.7, label=f"flip={flips[n_time]}, mean={mean_pair_corr:.2f}")
    #     plt.legend()
    #
    #     plt.xlabel("Pair Correlation")
    #     plt.ylabel("Frequency")
    #     plt.title("Pair Correlation Distribution")
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(dir_path, f"pair_corr_dist_{n_time}.png"))
    #     plt.close()

if __name__ == "__main__":
    main()