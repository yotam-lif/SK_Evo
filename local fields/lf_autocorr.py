import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from misc.Funcs import (
    init_alpha,
    init_h,
    init_J,
    relax_sk,
    calc_energies,
    compute_alpha_from_hist
)

def main():
    # Parameters
    N = 1000  # Number of spins
    random_state = 42  # Seed for reproducibility
    beta = 1.0
    rho = 1.0  # Sparsity of the coupling matrix
    num_stops = 5  # Number of stops for saved alphas

    # Initialize the model
    alpha_initial = init_alpha(N)
    h = init_h(N, random_state=random_state, beta=beta)
    J = init_J(N, random_state=random_state, beta=beta, rho=rho)

    # Perform relaxation
    flip_seq = relax_sk(alpha_initial.copy(), h, J, sswm=True)
    flip_num = len(flip_seq)
    flips = np.linspace(0, flip_num, num=num_stops, dtype=int)
    saved_alphas = [compute_alpha_from_hist(alpha_initial, flip_seq, flip) for flip in flips]

    # Create directory for saving plots
    output_dir = '../Plots/lf_autocorr_plots'
    os.makedirs(output_dir, exist_ok=True)

    alpha_0 = saved_alphas[0]
    # Iterate over flip_nums and plot scatter plots
    for i in range(len(saved_alphas) - 1):
        alpha_i = saved_alphas[i]
        alpha_i1 = saved_alphas[i + 1]

        # Calculate energies
        energies_i = calc_energies(alpha_i, h, J)
        energies_i1 = calc_energies(alpha_i1, h, J)
        energies_0 = calc_energies(alpha_0, h, J)

        # Calculate Pearson correlation
        corr_i_i1, _ = pearsonr(energies_i, energies_i1)
        corr_0_i1, _ = pearsonr(energies_0, energies_i1)

        # Plot scatter plots
        plt.figure(figsize=(12, 6))

        # Subplot 1: energies_i vs. energies_i1
        plt.subplot(1, 2, 1)
        plt.scatter(energies_i, energies_i1, alpha=0.5, edgecolor='k')
        plt.xlabel(f'Energies (flip {flips[i]})')
        plt.ylabel(f'Energies (flip {flips[i+1]})')
        plt.title(f'Pearson Correlation: {corr_i_i1:.2f}')
        max_val = max(max(energies_i), max(energies_i1))
        min_val = min(min(energies_i), min(energies_i1))
        # Plot y=x line
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        # Plot y=-x line
        plt.plot([min_val, max_val], [-min_val, max_val], 'r--')

        # Subplot 2: energies_0 vs. energies_i1
        plt.subplot(1, 2, 2)
        plt.scatter(energies_0, energies_i1, alpha=0.5, edgecolor='k')
        plt.xlabel(f'Energies (flip 0)')
        plt.ylabel(f'Energies (flip {flips[i+1]})')
        plt.title(f'Pearson Correlation: {corr_0_i1:.2f}')
        max_val = max(max(energies_0), max(energies_i1))
        min_val = min(min(energies_0), min(energies_i1))
        # Plot y=x line
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        # Plot y=max(y_value)-x line
        plt.plot([min_val, max_val], [max(energies_0)-min_val, max(energies_0)-max_val], 'r--')

        # Save the figure
        plot_filename = f'lf_autocorr_{i}.png'
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()

    print("Plots have been saved successfully.")

if __name__ == "__main__":
    main()