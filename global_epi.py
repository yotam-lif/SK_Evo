import os
import numpy as np
import matplotlib.pyplot as plt
from Funcs import (
    init_alpha,
    init_h,
    init_J,
    relax_SK,
    calc_DFE,
    compute_fit_slow,
    calc_rank,
    calc_F_off,
    calc_basic_lfs
)

def main():
    # Parameters
    N = 4000  # Number of spins
    random_state = 42  # Seed for reproducibility
    beta = 1.0  # Inverse temperature
    rho = 1.0  # Sparsity of the coupling matrix

    # Initialize the model
    alpha = init_alpha(N)
    h = init_h(N, random_state=random_state, beta=beta)
    J = init_J(N, random_state=random_state, beta=beta, rho=rho)
    print(f'Initial fitness = {compute_fit_slow(alpha, h, J):.4f}')

    # Uncomment the following line if F_off calculation is needed
    # F_off = calc_F_off(alpha, h, J)
    F_off = 0
    print(f'F_off = {F_off:.2f}')

    # Define ranks at which to save the configurations
    # Here, we choose to save at 30 equally spaced ranks from initial rank to 0
    initial_rank = calc_rank(alpha, h, J)
    num_saves = 30
    if initial_rank < num_saves:
        ranks_to_save = list(range(initial_rank, -1, -1))
    else:
        step = initial_rank // num_saves
        ranks_to_save = list(range(initial_rank, -1, -step))
        # Ensure that 0 is included
        if 0 not in ranks_to_save:
            ranks_to_save.append(0)

    # Remove duplicates and sort in descending order
    ranks_to_save = sorted(list(set(ranks_to_save)), reverse=True)

    print(f"Initial Rank: {initial_rank}")
    print(f"Ranks to Save: {ranks_to_save}")

    # Perform relaxation
    final_alpha, saved_alphas, saved_flips, saved_ranks = relax_SK(
        alpha=alpha.copy(),
        his=h,
        Jijs=J,
        ranks=ranks_to_save,
        sswm=True  # Change to False to use Glauber flip
    )

    # Initialize lists to store metrics
    average_DFE_list = []
    average_BDFE_list = []
    max_BDFE_list = []
    fitness_list = []
    rank_list = []  # Renamed from len_BDFE_list
    variance_DFE_list = []  # List to store variance of DFE
    variance_BDFE_list = []  # List to store variance of BDFE

    # Iterate over saved alphas and compute metrics
    for idx, saved_alpha in enumerate(saved_alphas):
        if saved_alpha is not None:
            DFE = calc_DFE(saved_alpha, h, J)
            BDFE = DFE[DFE > 0]

            # Compute average DFE
            average_DFE = np.mean(DFE)

            # Compute average BDFE, handle cases with no beneficial effects
            if BDFE.size > 0:
                average_BDFE = np.mean(BDFE)
                max_BDFE = np.max(BDFE)
                rank = BDFE.size  # Renamed from len_BDFE
            else:
                average_BDFE = np.nan
                max_BDFE = np.nan
                rank = 0  # Assign 0 if no beneficial effects

            # Compute fitness
            fitness = compute_fit_slow(saved_alpha, h, J, F_off=F_off)

            # Compute variance of DFE
            variance_DFE = np.var(DFE)
            variance_DFE_list.append(variance_DFE)

            # Compute variance of BDFE, handle cases with no beneficial effects
            if BDFE.size > 0:
                variance_BDFE = np.var(BDFE)
            else:
                variance_BDFE = np.nan
            variance_BDFE_list.append(variance_BDFE)

            # Append to lists
            average_DFE_list.append(average_DFE)
            average_BDFE_list.append(average_BDFE)
            max_BDFE_list.append(max_BDFE)
            fitness_list.append(fitness)
            rank_list.append(rank)  # Renamed from len_BDFE_list

            print(f"Saved Rank {ranks_to_save[idx]}: "
                  f"Average DFE = {average_DFE:.4f}, "
                  f"Average BDFE = {average_BDFE:.4f}, "
                  f"BDFE Size = {BDFE.size}, "
                  f"Max BDFE = {max_BDFE:.4f}, "
                  f"Fitness = {fitness:.4f}, "
                  f"Flips = {saved_flips[idx]}")
        else:
            print(f"Rank {ranks_to_save[idx]} was not reached during relaxation.")
            average_DFE_list.append(np.nan)
            average_BDFE_list.append(np.nan)
            max_BDFE_list.append(np.nan)
            fitness_list.append(np.nan)
            rank_list.append(np.nan)  # Assign NaN for rank
            variance_DFE_list.append(np.nan)  # Assign NaN for variance_DFE
            variance_BDFE_list.append(np.nan)  # Assign NaN for variance_BDFE

    # Convert lists to numpy arrays for plotting
    average_DFE_array = np.array(average_DFE_list)
    average_BDFE_array = np.array(average_BDFE_list)
    max_BDFE_array = np.array(max_BDFE_list)
    fitness_array = np.array(fitness_list)
    rank_array = np.array(rank_list)  # Renamed from len_BDFE_array
    variance_DFE_array = np.array(variance_DFE_list)  # Array for variance of DFE
    variance_BDFE_array = np.array(variance_BDFE_list)  # Array for variance of BDFE

    # Remove any NaN entries for the main metrics
    valid_main = (~np.isnan(fitness_array) &
                  ~np.isnan(average_DFE_array) &
                  ~np.isnan(average_BDFE_array) &
                  ~np.isnan(max_BDFE_array))
    fitness_main = fitness_array[valid_main]
    average_DFE_main = average_DFE_array[valid_main]
    average_BDFE_main = average_BDFE_array[valid_main]
    max_BDFE_main = max_BDFE_array[valid_main]

    # Remove any NaN entries for rank
    valid_rank = ~np.isnan(rank_array)
    fitness_rank = fitness_array[valid_rank]
    rank_valid = rank_array[valid_rank]

    # Remove any NaN entries for the variances
    valid_variance = (~np.isnan(variance_DFE_array) & ~np.isnan(variance_BDFE_array))
    fitness_variance = fitness_array[valid_variance]
    variance_DFE_valid = variance_DFE_array[valid_variance]
    variance_BDFE_valid = variance_BDFE_array[valid_variance]

    # Perform general linear fit: Average DFE = m * Fitness + b
    m_DFE, b_DFE = np.polyfit(fitness_main, average_DFE_main, 1)
    print(f"Fitted slope for Average DFE (m_DFE): {m_DFE:.4f}")
    print(f"Fitted intercept for Average DFE (b_DFE): {b_DFE:.4f}")

    # Perform general linear fit: Average BDFE = m * Fitness + b
    m_BDFE, b_BDFE = np.polyfit(fitness_main, average_BDFE_main, 1)
    print(f"Fitted slope for Average BDFE (m_BDFE): {m_BDFE:.4f}")
    print(f"Fitted intercept for Average BDFE (b_BDFE): {b_BDFE:.4f}")

    # Perform general linear fit: Max BDFE = m * Fitness + b
    m_maxBDFE, b_maxBDFE = np.polyfit(fitness_main, max_BDFE_main, 1)
    print(f"Fitted slope for Max BDFE (m_maxBDFE): {m_maxBDFE:.4f}")
    print(f"Fitted intercept for Max BDFE (b_maxBDFE): {b_maxBDFE:.4f}")

    # Perform linear fit for rank
    m_rank, b_rank = np.polyfit(fitness_rank, rank_valid, 1)
    print(f"Fitted slope for rank (m_rank): {m_rank:.4f}")
    print(f"Fitted intercept for rank (b_rank): {b_rank:.4f}")

    # Perform linear fit for variance of DFE
    m_var_DFE, b_var_DFE = np.polyfit(fitness_variance, variance_DFE_valid, 1)
    print(f"Fitted slope for Variance of DFE (m_var_DFE): {m_var_DFE:.4f}")
    print(f"Fitted intercept for Variance of DFE (b_var_DFE): {b_var_DFE:.4f}")

    # Perform linear fit for variance of BDFE
    m_var_BDFE, b_var_BDFE = np.polyfit(fitness_variance, variance_BDFE_valid, 1)
    print(f"Fitted slope for Variance of BDFE (m_var_BDFE): {m_var_BDFE:.4f}")
    print(f"Fitted intercept for Variance of BDFE (b_var_BDFE): {b_var_BDFE:.4f}")

    print(f'Slope for DFE * -N = {m_DFE * -N:.3f}')
    print(f'Slope for BDFE * -N = {m_BDFE * -N:.3f}')
    print(f'Slope for Max DFE * -N = {m_maxBDFE * -N:.3f}')

    print('Slope for variance of DFE * -N = {:.3f}'.format(m_var_DFE * -N))
    print('Slope for variance of BDFE * -N = {:.3f}'.format(m_var_BDFE * -N))

    # Generate fit line values
    fit_fitness_main = np.linspace(fitness_main.min(), fitness_main.max(), 500)
    fit_DFE = m_DFE * fit_fitness_main + b_DFE
    fit_BDFE = m_BDFE * fit_fitness_main + b_BDFE
    fit_maxBDFE = m_maxBDFE * fit_fitness_main + b_maxBDFE

    fit_fitness_rank = np.linspace(fitness_rank.min(), fitness_rank.max(), 500)
    fit_rank = m_rank * fit_fitness_rank + b_rank  # Fit line for rank

    fit_fitness_variance = np.linspace(fitness_variance.min(), fitness_variance.max(), 500)
    fit_var_DFE = m_var_DFE * fit_fitness_variance + b_var_DFE  # Fit line for variance of DFE
    fit_var_BDFE = m_var_BDFE * fit_fitness_variance + b_var_BDFE  # Fit line for variance of BDFE

    # Compute the theoretical variance curve
    sigma_h2 = np.var(h)
    sigma_J2 = np.var(J)
    theoretical_variance = 4 * (sigma_h2 + N*sigma_J2) - 4 * (fitness_array / N) ** 2

    # Create directory for saving plots
    output_dir = 'global_epi'
    os.makedirs(output_dir, exist_ok=True)

    # Plotting Metrics vs Fitness
    plt.figure(figsize=(14, 10))

    # Plot Average DFE
    plt.scatter(fitness_main, average_DFE_main, color='blue', label='Average DFE', alpha=0.6)
    plt.plot(fit_fitness_main, fit_DFE, color='blue', linestyle='--',
             label=f'Fit Average DFE: $<\\Delta_i(t)>$ = {m_DFE:.4f} * F(t) + {b_DFE:.4f}')

    # Plot Average BDFE
    plt.scatter(fitness_main, average_BDFE_main, color='green', label='Average BDFE', alpha=0.6)
    plt.plot(fit_fitness_main, fit_BDFE, color='green', linestyle='--',
             label=f'Fit Average BDFE: $<B\\Delta_i(t)>$ = {m_BDFE:.4f} * F(t) + {b_BDFE:.4f}')

    # Plot Max BDFE
    plt.scatter(fitness_main, max_BDFE_main, color='red', label='Max BDFE', alpha=0.6)
    plt.plot(fit_fitness_main, fit_maxBDFE, color='red', linestyle='--',
             label=f'Fit Max BDFE: max($B\\Delta_i(t)$) = {m_maxBDFE:.4f} * F(t) + {b_maxBDFE:.4f}')

    plt.xlabel('Fitness F(t)', fontsize=14)
    plt.ylabel('Fitness Metrics', fontsize=14)
    plt.title('SK Model Relaxation: DFE, BDFE Metrics vs Fitness', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_vs_fitness.png'))
    plt.show()

    # Plotting rank vs Fitness in a Separate Plot
    plt.figure(figsize=(14, 6))

    # Scatter plot for rank
    plt.scatter(fitness_rank, rank_valid, color='purple', label='rank', alpha=0.6)
    plt.plot(fit_fitness_rank, fit_rank, color='purple', linestyle='--',
             label=f'rank = {m_rank:.4f} * F(t) + {b_rank:.4f}')

    plt.xlabel('Fitness F(t)', fontsize=14)
    plt.ylabel('Number of Beneficial Fitness Effects (rank)', fontsize=14)
    plt.title('SK Model Relaxation: rank vs Fitness', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rank_vs_fitness.png'))
    plt.show()

    # Plotting Variance of DFE and BDFE vs Fitness
    plt.figure(figsize=(14, 6))

    # Scatter plot for variance of DFE
    plt.scatter(fitness_variance, variance_DFE_valid, color='blue', label='Variance of DFE', alpha=0.6)
    plt.plot(fit_fitness_variance, fit_var_DFE, color='blue', linestyle='--',
             label=f'Fit Variance of DFE: $\\sigma^2(\\Delta_i(t))$ = {m_var_DFE:.4f} * F(t) + {b_var_DFE:.4f}')

    # Scatter plot for variance of BDFE
    plt.scatter(fitness_variance, variance_BDFE_valid, color='green', label='Variance of BDFE', alpha=0.6)
    plt.plot(fit_fitness_variance, fit_var_BDFE, color='green', linestyle='--',
             label=f'Fit Variance of BDFE: $\\sigma^2(B\\Delta_i(t))$ = {m_var_BDFE:.4f} * F(t) + {b_var_BDFE:.4f}')

    # Plot theoretical variance curve
    plt.plot(fitness_array, theoretical_variance, color='orange', linestyle='-', label=r'Theoretical $\langle \Delta_i^2 \rangle$')

    plt.xlabel('Fitness F(t)', fontsize=14)
    plt.ylabel('Variance', fontsize=14)
    plt.title('Variance of DFE and BDFE vs Fitness', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variance_vs_fitness.png'))
    plt.show()

if __name__ == "__main__":
    main()