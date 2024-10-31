import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr  # For calculating correlation coefficients
import pandas as pd  # For saving correlation data

# Import the Funcs module
from Funcs import *

def main():
    # -------------------------------
    # 1. Initialize SK Environment
    # -------------------------------

    # Parameters
    N_start = 1000
    N_end = 2000
    N_step = 200
    N_num = int((N_end - N_start) / N_step) + 1
    N = np.linspace(N_start, N_end, N_num, dtype=int)  # Number of spins
    beta = 1.0  # Epistasis strength
    rho = 1.0  # Fraction of non-zero coupling elements
    random_state = 42  # Seed for reproducibility
    num_saves = 30

    plt.figure()
    plt.xlabel("Rank / N")
    plt.ylabel(f"$ N\\langle \\Delta_{{ij}} \\rangle $ for unstable $i$")
    plt.title("Mean Drifts vs Rank")
    plt.gca().invert_xaxis()  # Reverse the x-axis

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dir_path = os.path.join(dir_path, "Plots", "drift_means")
    os.makedirs(dir_path, exist_ok=True)

    for n in N:
        # Initialize the model
        alpha = init_alpha(n)
        h = init_h(n, random_state=random_state, beta=beta)
        J = init_J(n, random_state=random_state, beta=beta, rho=rho)

        # Define ranks at which to save the configurations
        initial_rank = calc_rank(alpha, h, J)
        ranks_to_save = np.linspace(0, initial_rank, num_saves, dtype=int)
        # Reverse the order of the ranks to save
        ranks_to_save = ranks_to_save[::-1]
        print(f"Initial Rank: {initial_rank}")

        # Perform relaxation
        final_alpha, saved_alphas = relax_sk_ranks(
            alpha=alpha.copy(),
            his=h,
            Jijs=J,
            ranks=ranks_to_save,
            sswm=True  # Change to False to use Glauber flip
            )

        # -------------------------------
        # 2. Calculate Drifts
        # -------------------------------
        # We want to look at the terms in deltas, Delta_ij.
        # We want to look at the mean of these terms for i's, such that Delta_i > 0, i.e. unstable spins.
        means = []
        valid_ranks = []
        for rank, alpha_i in zip(ranks_to_save, saved_alphas):
            if alpha_i is not None:
                Delta_ij = np.outer(alpha_i, alpha_i)
                Delta_ij = -2 * np.multiply(Delta_ij, J)
                bdfe, bdfe_ind = calc_BDFE(alpha_i, h, J)
                # Get Delta_ij with rows only corresponding to indexes in bdfe_ind
                Delta_ij = Delta_ij[bdfe_ind]
                # Get the mean of all terms
                mean_Delta_ij = np.mean(Delta_ij, axis=1)
                means.append(mean_Delta_ij)
                valid_ranks.append(rank)

        valid_ranks = np.array(valid_ranks) / n
        means = np.array(means) * n
        plt.plot(valid_ranks, means, marker='o', label=f"N={int(n)}")

    # -------------------------------
    # 3. Plot Drifts
    # -------------------------------
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(dir_path, "drift_means.png"))

if __name__ == "__main__":
    main()