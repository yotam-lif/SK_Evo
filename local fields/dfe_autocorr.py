import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns
from cmn.cmn_sk import (
    init_alpha,
    init_h,
    init_J,
    relax_sk,
    compute_dfe,
    compute_alpha_from_hist
)

def main():
    # Parameters
    N = 2000  # Number of spins
    random_state = 42  # Seed for reproducibility
    beta = 1.0
    rho = 1.0  # Sparsity of the coupling matrix
    num_stops = 20  # Number of stops for saved alphas

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
    output_dir = '../Plots/dfe_autocorr_plots'
    os.makedirs(output_dir, exist_ok=True)
    scatter_dir = os.path.join(output_dir, 'scatter_plots')
    ridge_dir = os.path.join(output_dir, 'ridge_plots')
    os.makedirs(scatter_dir, exist_ok=True)
    os.makedirs(ridge_dir, exist_ok=True)

    alpha_0 = saved_alphas[0]
    # Iterate over flip_nums and plot scatter plots
    for i in range(len(saved_alphas) - 1):
        alpha_i = saved_alphas[i]
        alpha_i1 = saved_alphas[i + 1]

        # Calculate energies
        deltas_i = compute_dfe(alpha_i, h, J)
        deltas_i1 = compute_dfe(alpha_i1, h, J)
        deltas_0 = compute_dfe(alpha_0, h, J)

        # Calculate Pearson correlation
        corr_i_i1, _ = pearsonr(deltas_i, deltas_i1)
        corr_0_i1, _ = pearsonr(deltas_0, deltas_i1)

        # Plot scatter plots
        plt.figure(figsize=(12, 6))

        # Subplot 1: energies_i vs. energies_i1
        plt.subplot(1, 2, 1)
        plt.scatter(deltas_i, deltas_i1, alpha=0.5, edgecolor='k')
        plt.xlabel(f'DFE (flip {flips[i]})')
        plt.ylabel(f'DFE (flip {flips[i+1]})')
        plt.title(f'Pearson Correlation: {corr_i_i1:.2f}')
        max_val = max(max(deltas_i), max(deltas_i1))
        min_val = min(min(deltas_i), min(deltas_i1))
        # Plot y=x line
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        # Subplot 2: energies_0 vs. energies_i1
        plt.subplot(1, 2, 2)
        plt.scatter(deltas_0, deltas_i1, alpha=0.5, edgecolor='k')
        plt.xlabel(f'DFE (flip 0)')
        plt.ylabel(f'DFE (flip {flips[i+1]})')
        plt.title(f'Pearson Correlation: {corr_0_i1:.2f}')

        # Save the figure
        plot_filename = f'dfe_autocorr_{i}.png'
        plt.tight_layout()
        plt.savefig(os.path.join(scatter_dir, plot_filename))
        plt.close()

        # Plot ridge plots
        plt.figure(figsize=(12, 6))
        # Subplot 1: energies_i vs. energies_i1
        plt.subplot(1, 2, 1)
        # Create a DataFrame
        df = pd.DataFrame({
            f'DFE (flip {flips[i]})': deltas_i,
            f'DFE (flip {flips[i+1]})': deltas_i1
        })
        sns.kdeplot(data=df, x=f'DFE (flip {flips[i]})', y=f'DFE (flip {flips[i+1]})', fill=True)
        plt.title(f'Pearson Correlation: {corr_i_i1:.2f}')
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        # Subplot 2: energies_0 vs. energies_i1
        plt.subplot(1, 2, 2)
        # Create a DataFrame
        df = pd.DataFrame({
            f'DFE (flip 0)': deltas_0,
            f'DFE (flip {flips[i+1]})': deltas_i1
        })
        sns.kdeplot(data=df, x=f'DFE (flip 0)', y=f'DFE (flip {flips[i+1]})', fill=True)
        plt.title(f'Pearson Correlation: {corr_0_i1:.2f}')

        # Save the figure
        plot_filename = f'dfe_autocorr_{i}.png'
        plt.tight_layout()
        plt.savefig(os.path.join(ridge_dir, plot_filename))
        plt.close()

    print("Plots have been saved successfully.")

if __name__ == "__main__":
    main()