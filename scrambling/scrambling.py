import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from misc import cmn as Fs


def plot_dfe_comparison(ax: plt.Axes, dfe_current, dfe_propagated, bins, title):
    """
    Plot the total dfe and the propagated dfe on the same axis.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axis to plot on.
    dfe_current : np.ndarray
        The dfe at the current time t.
    dfe_propagated : np.ndarray
        The propagated dfe from max rank to time t.
    bins : int
        Number of bins for histograms.
    title : str
        Title for the plot.
    """
    sns.histplot(dfe_current, bins=bins, kde=False, stat="probability", color="purple",
                 label='Current dfe', ax=ax, alpha=0.7)
    sns.histplot(dfe_propagated, bins=bins, kde=False, stat="probability", color="blue",
                 label='Propagated dfe', ax=ax, alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel('Fitness effect')
    ax.set_ylabel('Frequency')
    ax.legend()


def main():
    N = 1000  # Number of spins
    beta = 0.25  # Epistasis strength
    rho = 0.05  # Fraction of non-zero coupling elements
    bins = 60

    # Initialize spin configuration
    alpha_initial = Fs.init_alpha(N)

    # Initialize external fields
    h = Fs.init_h(N, beta=beta)

    # Initialize coupling matrix with sparsity
    J = Fs.init_J(N, beta=beta, rho=rho)

    # Define desired ranks
    min_rank = 0
    max_rank = 500
    num_ranks = 10
    ranks = np.linspace(min_rank, max_rank, num_ranks, dtype=int)
    ranks = sorted(ranks, reverse=True)  # [500, 444, ..., 0]

    # Relax the system using sswm_flip (sswm=True)
    final_alpha, saved_alphas, flips, saved_ranks = Fs.relax_SK(alpha_initial.copy(), h, J, ranks, sswm=True)

    # Create main output directory for plots
    base_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.join(base_dir, "dfe_plots")
    os.makedirs(main_dir, exist_ok=True)

    # Debugging: Check which alphas were saved
    print("\n--- Saved Alphas Check ---")
    for rank, alpha in zip(ranks, saved_alphas):
        if alpha is not None:
            print(f"Rank {rank}: Alpha saved.")
        else:
            print(f"Rank {rank}: Alpha NOT saved.")
    print("--------------------------\n")

    # Collect DFEs and times
    dfes = [Fs.calc_DFE(alpha, h, J) if alpha is not None else None for alpha in saved_alphas]
    times = flips  # Number of mutations up to each saved rank

    # Identify dfe at max_rank (assuming it's the first one)
    dfe_max_rank = dfes[0]
    if dfe_max_rank is None:
        print("dfe at max_rank is not available. Exiting.")
        return

    # Iterate over ranks and corresponding saved alphas
    for i, (rank, alpha_t, dfe_t, t) in enumerate(zip(ranks, saved_alphas, dfes, times)):
        if alpha_t is not None and dfe_t is not None:
            print(f"Processing Rank {rank} with {t} mutations.")

            # Forward propagate: from max_rank dfe to current dfe
            # We need to propagate the dfe from max_rank to rank t
            # For total dfe, we consider all mutations

            # Since mutations correspond to flipping spins, and the dfe represents the fitness effects
            # of flipping each spin, we can use the indices where mutations have occurred

            # Find the indices of spins that have flipped between max_rank and current rank
            flips_between_ranks = np.where(alpha_t != saved_alphas[0])[0]

            # Propagate the dfe from max_rank to time t
            propagated_dfe = dfe_max_rank.copy()
            # Update the dfe at the mutated sites
            for idx in flips_between_ranks:
                # Recalculate the fitness effect at this site in the current configuration
                delta_fit = Fs.compute_fitness_delta_mutant(alpha_t, h, J, idx)
                propagated_dfe[idx] = delta_fit

            # Plotting
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot the total dfe at time t and the propagated dfe
            plot_dfe_comparison(
                ax=ax,
                dfe_current=dfe_t,
                dfe_propagated=propagated_dfe,
                bins=bins,
                title=f"dfe Comparison at Rank {rank}"
            )

            plt.suptitle(f"Rank {rank} with {t} mutations", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(main_dir, f"dfe_comparison_rank_{rank}_nmuts_{t}.png"), dpi=300)
            plt.close()

    print("Plotting completed successfully.")


if __name__ == '__main__':
    main()