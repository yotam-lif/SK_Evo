# relaxation_script.py

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
    calc_basic_lfs  # Newly imported function
)


def main():
    # Parameters
    N = 3000  # Number of spins
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
    sig_J = np.sqrt(beta / (N * rho))
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
    final_alpha, saved_alphas, saved_flips = relax_SK(
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
    mean_abs_lfs_list = []  # New list to store mean absolute local fields

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

            # Calculate basic local fields
            basic_lfs = calc_basic_lfs(saved_alpha, h, J)
            abs_basic_lfs = np.abs(basic_lfs)
            mean_abs_lfs = np.mean(abs_basic_lfs)
            mean_abs_lfs_list.append(mean_abs_lfs)

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
                  f"Mean Abs Local Fields = {mean_abs_lfs:.4f}, "
                  f"Flips = {saved_flips[idx]}")
        else:
            print(f"Rank {ranks_to_save[idx]} was not reached during relaxation.")
            average_DFE_list.append(np.nan)
            average_BDFE_list.append(np.nan)
            max_BDFE_list.append(np.nan)
            fitness_list.append(np.nan)
            rank_list.append(np.nan)  # Assign NaN for rank
            mean_abs_lfs_list.append(np.nan)  # Assign NaN for mean_abs_lfs

    # Convert lists to numpy arrays for plotting
    average_DFE_array = np.array(average_DFE_list)
    average_BDFE_array = np.array(average_BDFE_list)
    max_BDFE_array = np.array(max_BDFE_list)
    fitness_array = np.array(fitness_list)
    rank_array = np.array(rank_list)  # Renamed from len_BDFE_array
    mean_abs_lfs_array = np.array(mean_abs_lfs_list)  # Array for mean absolute local fields

    # Remove any NaN entries for the main metrics
    valid_main = (~np.isnan(fitness_array) &
                  ~np.isnan(average_DFE_array) &
                  ~np.isnan(average_BDFE_array) &
                  ~np.isnan(max_BDFE_array))
    fitness_main = fitness_array[valid_main]
    average_DFE_main = average_DFE_array[valid_main]
    average_BDFE_main = average_BDFE_array[valid_main]
    max_BDFE_main = max_BDFE_array[valid_main]
    mean_abs_lfs_main = mean_abs_lfs_array[valid_main]  # Corresponding mean abs local fields

    # Remove any NaN entries for rank
    valid_rank = ~np.isnan(rank_array)
    fitness_rank = fitness_array[valid_rank]
    rank_valid = rank_array[valid_rank]

    # Remove any NaN entries for mean_abs_lfs
    valid_mean_abs_lfs = ~np.isnan(mean_abs_lfs_array)
    fitness_lfs = fitness_array[valid_mean_abs_lfs]
    mean_abs_lfs_valid = mean_abs_lfs_array[valid_mean_abs_lfs]

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

    # Perform linear fit for mean absolute local fields
    m_mean_abs_lfs, b_mean_abs_lfs = np.polyfit(fitness_lfs, mean_abs_lfs_valid, 1)
    print(f"Fitted slope for Mean Abs Local Fields (m_mean_abs_lfs): {m_mean_abs_lfs:.4f}")
    print(f"Fitted intercept for Mean Abs Local Fields (b_mean_abs_lfs): {b_mean_abs_lfs:.4f}")

    print(f'Slope for BDFE * -N = {m_BDFE * -N:.3f}')
    print(f'Slope for Max DFE * -N = {m_maxBDFE * -N:.3f}')

    # Generate fit line values
    fit_fitness_main = np.linspace(fitness_main.min(), fitness_main.max(), 500)
    fit_DFE = m_DFE * fit_fitness_main + b_DFE
    fit_BDFE = m_BDFE * fit_fitness_main + b_BDFE
    fit_maxBDFE = m_maxBDFE * fit_fitness_main + b_maxBDFE

    fit_fitness_rank = np.linspace(fitness_rank.min(), fitness_rank.max(), 500)
    fit_rank = m_rank * fit_fitness_rank + b_rank  # Fit line for rank

    fit_fitness_lfs = np.linspace(fitness_lfs.min(), fitness_lfs.max(), 500)
    fit_mean_abs_lfs = m_mean_abs_lfs * fit_fitness_lfs + b_mean_abs_lfs  # Fit line for mean abs local fields

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
    plt.show()

    # # Plotting Mean Absolute Local Fields vs Fitness in a Separate Plot
    # plt.figure(figsize=(14, 6))
    #
    # # Scatter plot for mean absolute local fields
    # plt.scatter(fitness_lfs, mean_abs_lfs_valid, color='orange', label='Mean Abs Local Fields', alpha=0.6)
    # plt.plot(fit_fitness_lfs, fit_mean_abs_lfs, color='orange', linestyle='--',
    #          label=f'Mean Abs Local Fields = {m_mean_abs_lfs:.4f} * F(t) + {b_mean_abs_lfs:.4f}')
    #
    # plt.xlabel('Fitness F(t)', fontsize=14)
    # plt.ylabel('Mean Absolute Local Fields', fontsize=14)
    # plt.title('SK Model Relaxation: Mean Absolute Local Fields vs Fitness', fontsize=16)
    # plt.legend(fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()