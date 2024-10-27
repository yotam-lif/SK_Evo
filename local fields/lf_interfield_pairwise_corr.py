import os
import numpy as np
import matplotlib.pyplot as plt
from Funcs import (
    init_alpha,
    init_h,
    init_J,
    relax_SK,
)

def main():
    # Parameters
    N = 1000  # Number of spins
    random_state = 42  # Seed for reproducibility
    beta = 1.0
    rho = 1.0  # Sparsity of the coupling matrix
    # Define the number of lowest ranks to plot
    n_times = 5  # Example value, adjust as needed
    n_bins = 50
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dir_path = os.path.join(dir_path, "Plots", "pair_corr_dist")
    os.makedirs(dir_path, exist_ok=True)

    # Initialize the model
    alpha = init_alpha(N)
    h = init_h(N, random_state=random_state, beta=beta)
    J = init_J(N, random_state=random_state, beta=beta, rho=rho)
    flips = np.linspace(0, N * 0.6, n_times, dtype=int)

    final_alpha, saved_alphas, saved_flips, saved_ranks = relax_SK(
        alpha=alpha.copy(),
        his=h,
        Jijs=J,
        flips=flips,
        sswm=True  # Change to False to use Glauber flip
    )

    pair_corr_list = np.zeros((n_times, N))
    for n_time, flip in enumerate(flips):
        alpha_i = saved_alphas[n_time]
        for i in range(N):
            pair_corr = 0
            for m in range(N):
                for n in range(N):
                    if m != n:
                        pair_corr += J[i, m] * J[i, n] * alpha_i[m] * alpha_i[n]
            pair_corr_list[n_time, i] = pair_corr

    pair_corr_list /= N
    # now for n_times, we plot the distribution of pair correlations as histogram
    for n_time in range(n_times):
        plt.figure()
        plt.hist(pair_corr_list[n_time], bins=n_bins, alpha=0.5, label=f"flip={flips[n_time]}")
        plt.legend()
        plt.xlabel("Pair Correlation")
        plt.ylabel("Frequency")
        plt.title("Pair Correlation Distribution")
        plt.savefig(os.path.join(dir_path, f"pair_corr_dist_{n_time}.png"))
        plt.close()

if __name__ == "__main__":
    main()