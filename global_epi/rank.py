import os
import numpy as np
import matplotlib.pyplot as plt
from misc.cmn_sk import (
    init_alpha,
    init_h,
    init_J,
    relax_sk_ranks,
    compute_fit_slow,
    calc_rank,
    calc_DFE
)

def main():
    # Parameters
    N = 1000
    random_state = 42
    beta = 1.0
    rho = 1.0
    num_saves = 300

    # Create directory for saving plots
    output_dir = '../Plots/global_epi'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the model
    alpha = init_alpha(N)
    h = init_h(N, random_state=random_state, beta=beta)
    J = init_J(N, random_state=random_state, beta=beta, rho=rho)

    # Define ranks at which to save the configurations
    initial_rank = calc_rank(alpha, h, J)
    ranks_to_save = np.linspace(int(N / 10), initial_rank, num_saves, dtype=int)
    # make sure it is sorted in descending order
    ranks_to_save = sorted(list(set(ranks_to_save)), reverse=True)

    # Perform relaxation
    final_alpha, saved_alphas = relax_sk_ranks(
        alpha=alpha.copy(),
        his=h,
        Jijs=J,
        ranks=ranks_to_save,
        sswm=True
    )

    # Initialize lists to store metrics
    rank_list = []
    fitness_list = []

    # Iterate over saved alphas and compute metrics
    for saved_alpha in saved_alphas:
        if saved_alpha is not None:
            DFE = calc_DFE(saved_alpha, h, J)
            BDFE = DFE[DFE > 0]
            rank = len(BDFE)
            fitness = compute_fit_slow(saved_alpha, h, J)
            rank_list.append(rank)
            fitness_list.append(fitness)

    # Convert lists to numpy arrays for plotting
    rank_array = np.array(rank_list)
    fitness_array = np.array(fitness_list)

    # Remove any NaN entries
    valid = ~np.isnan(rank_array)
    fitness_valid = fitness_array[valid]
    rank_valid = rank_array[valid]

    # Perform linear fit
    m_rank, b_rank = np.polyfit(fitness_valid, rank_valid, 1)

    # Generate fit line values
    fit_fitness = np.linspace(fitness_valid.min(), fitness_valid.max(), 500)
    fit_rank = m_rank * fit_fitness + b_rank

    # Plotting Rank vs Fitness
    plt.figure(figsize=(14, 10))
    plt.scatter(fitness_valid, rank_valid, color='purple', label='Rank', alpha=0.6)
    plt.plot(fit_fitness, fit_rank, color='purple', linestyle='--',
             label=f'Fit Rank: rank = {m_rank:.4f} * F(t) + {b_rank:.4f}')
    plt.xlabel('Fitness F(t)', fontsize=14)
    plt.ylabel('Rank', fontsize=14)
    plt.title('Rank vs Fitness', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rank_vs_fitness.png'))
    plt.show()

if __name__ == "__main__":
    main()