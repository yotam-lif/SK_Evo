import os
import numpy as np
import matplotlib.pyplot as plt
from misc.cmn_sk import (
    init_h,
    init_J,
    relax_sk_ranks,
    compute_fit_slow,
    compute_rank,
    compute_bdfe
)
from misc.cmn import init_sigma

def main():
    # Parameters
    N = 2500
    random_state = 42
    beta = 1.0
    rho = 1.0
    num_saves = 300

    # Create directory for saving plots
    output_dir = '../Plots/global_epi'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the model
    alpha = init_sigma(N)
    h = init_h(N, random_state=random_state, beta=beta)
    J = init_J(N, random_state=random_state, beta=beta, rho=rho)

    # Define ranks at which to save the configurations
    initial_rank = compute_rank(alpha, h, J)
    ranks_to_save = np.linspace(int(N/8), initial_rank, num_saves, dtype=int)
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
    average_BDFE_list = []
    fitness_list = []

    # Iterate over saved alphas and compute metrics
    for saved_alpha in saved_alphas:
        if saved_alpha is not None:
            BDFE, _ = compute_bdfe(saved_alpha, h, J)
            average_BDFE = np.mean(BDFE) if len(BDFE) > 0 else np.nan
            average_BDFE_list.append(average_BDFE)
            fitness_list.append(compute_fit_slow(saved_alpha, h, J))

    # Convert lists to numpy arrays for plotting
    average_BDFE_array = np.array(average_BDFE_list)
    fitness_array = np.array(fitness_list)

    # Remove any NaN entries
    valid = ~np.isnan(average_BDFE_array)
    fitness_valid = fitness_array[valid]
    average_BDFE_valid = average_BDFE_array[valid]

    # Perform linear fit
    m_BDFE, b_BDFE = np.polyfit(fitness_valid, average_BDFE_valid, 1)

    # Generate fit line values
    fit_fitness = np.linspace(fitness_valid.min(), fitness_valid.max(), 500)
    fit_BDFE = m_BDFE * fit_fitness + b_BDFE

    # Plotting Average BDFE vs Fitness
    plt.figure(figsize=(14, 10))
    plt.scatter(fitness_valid, average_BDFE_valid, color='green', label='Average BDFE', alpha=0.6)
    plt.plot(fit_fitness, fit_BDFE, color='green', linestyle='--',
             label=f'Fit Average BDFE: $<B\\Delta_i(t)>$ = {m_BDFE:.5f} * F(t) + {b_BDFE:.4f}')
    plt.xlabel('Fitness F(t)', fontsize=14)
    plt.ylabel('Average BDFE', fontsize=14)
    plt.title(f'Average BDFE vs Fitness; N={N}', fontsize=16)
    plt.text(0.5, 0.5, f'm * N = {m_BDFE * N: .2f}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_BDFE_vs_fitness.png'))
    plt.show()

if __name__ == "__main__":
    main()