import os
import numpy as np
import matplotlib.pyplot as plt
from code.cmn.cmn_sk import (
    init_alpha,
    init_h,
    init_J,
    relax_sk_ranks,
    compute_fit_slow,
    compute_rank,
    compute_lfs, compute_energies
)

def main():
    # Parameters
    N = 1000  # Number of spins
    random_state = 42  # Seed for reproducibility
    beta = 1.0
    rho = 1.0  # Sparsity of the coupling matrix
    # Define the number of lowest ranks to plot
    n = 28  # Example value, adjust as needed
    bins = 50
    num_saves = 30

    # Initialize the model
    alpha = init_alpha(N)
    h = init_h(N, random_state=random_state, beta=beta)
    J = init_J(N, random_state=random_state, beta=beta, rho=rho)
    initial_fitness = compute_fit_slow(alpha, h, J)
    print(f'Initial fitness = {initial_fitness:.4f}')

    # Define ranks at which to save the configurations
    initial_rank = compute_rank(alpha, h, J)
    ranks_to_save = np.linspace(0, initial_rank, num_saves, dtype=int)
    # Reverse the order of the ranks to save
    ranks_to_save = ranks_to_save[::-1]
    print(f"Initial Rank: {initial_rank}")
    print(f"Ranks to Save: {ranks_to_save}")

    # Perform relaxation
    final_alpha, saved_alphas = relax_sk_ranks(
        alpha=alpha.copy(),
        his=h,
        Jijs=J,
        ranks=ranks_to_save,
        sswm=True  # Change to False to use Glauber flip
    )

    # Subset of ranks to plot local field distributions
    ranks_to_plot = ranks_to_save[-n:]  # Get the n lowest ranks

    # Create directory for saving plots
    output_dir = '../../plots/lf_dist_plots'
    os.makedirs(output_dir, exist_ok=True)

    # Plot histograms for all local fields and non-aligned local fields
    for rank, alpha_i in zip(ranks_to_plot, saved_alphas[-n:]):
        lfs = compute_lfs(alpha_i, h, J)
        eis = compute_energies(alpha_i, h, J)
        # Now take only the lfs of index i where eis[i] < 0
        aa_lfs = [lfs[i] for i in range(len(eis)) if eis[i] < 0]

        plt.figure(figsize=(12, 6))

        # Histogram for all local fields
        # plt.subplot(1, 2, 2)
        # plt.hist(lfs, bins=bins, alpha=0.7, label='All Local Fields', color='blue', edgecolor='black')
        # plt.xlabel('Local Field')
        # plt.ylabel('Frequency')
        # plt.title(f'All Local Fields Histogram (Rank {rank})')
        # plt.legend()

        # Histogram for non-aligned local fields
        plt.subplot(1, 2, 2)
        plt.hist(lfs, bins=bins, alpha=0.7, label='All Local Fields', color='blue', edgecolor='black')
        plt.hist(aa_lfs, bins=bins, alpha=0.7, label='Non-Aligned Local Fields', color='orange', edgecolor='black')
        plt.xlabel('Local Field')
        plt.ylabel('Frequency')
        plt.title(f'Non-Aligned Local Fields Histogram (Rank {rank})')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'lf_histogram_rank_{rank}.png'))
        plt.close()

if __name__ == "__main__":
    main()