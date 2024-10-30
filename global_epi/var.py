import os
import numpy as np
import matplotlib.pyplot as plt
from Funcs import (
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
    variance_DFE_list = []
    variance_BDFE_list = []
    fitness_list = []

    # Iterate over saved alphas and compute metrics
    for saved_alpha in saved_alphas:
        if saved_alpha is not None:
            DFE = calc_DFE(saved_alpha, h, J)
            BDFE = DFE[DFE > 0]
            variance_DFE = np.var(DFE)
            variance_BDFE = np.var(BDFE) if len(BDFE) > 0 else np.nan
            fitness = compute_fit_slow(saved_alpha, h, J)
            variance_DFE_list.append(variance_DFE)
            variance_BDFE_list.append(variance_BDFE)
            fitness_list.append(fitness)

    # Convert lists to numpy arrays for plotting
    variance_DFE_array = np.array(variance_DFE_list)
    variance_BDFE_array = np.array(variance_BDFE_list)
    fitness_array = np.array(fitness_list)

    # Remove any NaN entries
    valid = (~np.isnan(variance_DFE_array) & ~np.isnan(variance_BDFE_array))
    fitness_valid = fitness_array[valid]
    variance_DFE_valid = variance_DFE_array[valid]
    variance_BDFE_valid = variance_BDFE_array[valid]

    # Perform linear fit for variance of DFE
    m_var_DFE, b_var_DFE = np.polyfit(fitness_valid, variance_DFE_valid, 1)

    # Perform linear fit for variance of BDFE
    m_var_BDFE, b_var_BDFE = np.polyfit(fitness_valid, variance_BDFE_valid, 1)

    # Generate fit line values
    fit_fitness = np.linspace(fitness_valid.min(), fitness_valid.max(), 500)
    fit_var_DFE = m_var_DFE * fit_fitness + b_var_DFE
    fit_var_BDFE = m_var_BDFE * fit_fitness + b_var_BDFE


    # Plotting Variance of DFE and BDFE vs Fitness
    plt.figure(figsize=(14, 10))
    plt.scatter(fitness_valid, variance_DFE_valid, color='blue', label='Variance of DFE', alpha=0.6)
    plt.plot(fit_fitness, fit_var_DFE, color='blue', linestyle='--',
             label=f'Fit Variance of DFE: $\\sigma^2(\\Delta_i(t))$ = {m_var_DFE:.4f} * F(t) + {b_var_DFE:.4f}')
    plt.scatter(fitness_valid, variance_BDFE_valid, color='green', label='Variance of BDFE', alpha=0.6)
    plt.plot(fit_fitness, fit_var_BDFE, color='green', linestyle='--',
             label=f'Fit Variance of BDFE: $\\sigma^2(B\\Delta_i(t))$ = {m_var_BDFE:.4f} * F(t) + {b_var_BDFE:.4f}')
    plt.xlabel('Fitness F(t)', fontsize=14)
    plt.ylabel('Variance', fontsize=14)
    plt.title('Variance of DFE and BDFE vs Fitness', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.text(0.5, 0.5, f"m_DFE * N = {m_var_DFE * N:.4f}, m_BDFE * N = {m_var_BDFE * N:.4f}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variance_vs_fitness.png'))
    plt.show()

if __name__ == "__main__":
    main()