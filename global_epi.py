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
    calc_F_off
)
from scipy.stats import linregress


def main():
    # Parameters
    N = 2000  # Number of spins
    random_state = 42  # Seed for reproducibility
    beta = 1.0  # Inverse temperature
    rho = 1.0  # Sparsity of the coupling matrix

    # Initialize the model
    alpha = init_alpha(N)
    h = init_h(N, random_state=random_state, beta=beta)
    J = init_J(N, random_state=random_state, beta=beta, rho=rho)
    F_off = calc_F_off(alpha, h, J)

    # Compute initial fitness before relaxation
    initial_fitness = compute_fit_slow(alpha, h, J, F_off=F_off)
    print(f"Initial Fitness: {initial_fitness:.4f}")

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
    zero_bin_counts = []  # Track bins containing or near 0

    # Define threshold for considering a bin as "containing 0"
    # You can adjust this threshold as needed
    zero_threshold = 1e-8  # Essentially the bin that contains 0

    # Iterate over saved alphas and compute metrics
    for idx, saved_alpha in enumerate(saved_alphas):
        if saved_alpha is not None:
            DFE = calc_DFE(saved_alpha, h, J)

            # Compute Zero Bin Count
            # Create histogram with 50 bins
            num_bins = 50
            hist, bin_edges = np.histogram(DFE, bins=num_bins)

            # Find the bin that contains 0
            # Bin edges are sorted in ascending order
            # Find the index where bin_edges[i] <= 0 < bin_edges[i+1]
            bin_index = np.searchsorted(bin_edges, 0, side='right') - 1
            bin_index = np.clip(bin_index, 0, num_bins - 1)  # Ensure within range
            zero_bin_count = hist[bin_index]
            zero_bin_counts.append(zero_bin_count)

            # Calculate beneficial DFE (BDFE)
            BDFE = DFE[DFE > 0]

            # Compute average DFE
            average_DFE = np.mean(DFE)

            # Compute average BDFE, handle cases with no beneficial effects
            if BDFE.size > 0:
                average_BDFE = np.mean(BDFE)
                max_BDFE = np.max(BDFE)
            else:
                average_BDFE = np.nan
                max_BDFE = np.nan

            # Compute fitness
            fitness = compute_fit_slow(saved_alpha, h, J, F_off=F_off)

            # Append to lists
            average_DFE_list.append(average_DFE)
            average_BDFE_list.append(average_BDFE)
            max_BDFE_list.append(max_BDFE)
            fitness_list.append(fitness)

            print(f"Saved Rank {ranks_to_save[idx]}: "
                  f"Average DFE = {average_DFE:.4f}, "
                  f"Average BDFE = {average_BDFE:.4f}, "
                  f"Max BDFE = {max_BDFE:.4f}, "
                  f"Fitness = {fitness:.4f}, "
                  f"Zero Bin Count = {zero_bin_count}, "
                  f"Flips = {saved_flips[idx]}")
        else:
            print(f"Rank {ranks_to_save[idx]} was not reached during relaxation.")
            average_DFE_list.append(np.nan)
            average_BDFE_list.append(np.nan)
            max_BDFE_list.append(np.nan)
            fitness_list.append(np.nan)
            zero_bin_counts.append(np.nan)

    # Convert lists to numpy arrays for plotting
    average_DFE_array = np.array(average_DFE_list)
    average_BDFE_array = np.array(average_BDFE_list)
    max_BDFE_array = np.array(max_BDFE_list)
    fitness_array = np.array(fitness_list)
    zero_bin_counts_array = np.array(zero_bin_counts)

    # Remove any NaN entries
    valid_indices = (~np.isnan(fitness_array) &
                     ~np.isnan(average_DFE_array) &
                     ~np.isnan(average_BDFE_array) &
                     ~np.isnan(max_BDFE_array) &
                     ~np.isnan(zero_bin_counts_array))

    fitness_array = fitness_array[valid_indices]
    average_DFE_array = average_DFE_array[valid_indices]
    average_BDFE_array = average_BDFE_array[valid_indices]
    max_BDFE_array = max_BDFE_array[valid_indices]
    zero_bin_counts_array = zero_bin_counts_array[valid_indices]

    # Perform linear fits using np.polyfit for Average DFE, BDFE, Max BDFE
    # and using linregress for Zero Bin Count
    # This ensures consistency and provides slope and intercept with statistics

    # Fit for Average DFE
    m_DFE, b_DFE = np.polyfit(fitness_array, average_DFE_array, 1)
    print(f"Fitted slope for Average DFE (m_DFE): {m_DFE:.4f}")
    print(f"Fitted intercept for Average DFE (b_DFE): {b_DFE:.4f}")

    # Fit for Average BDFE
    m_BDFE, b_BDFE = np.polyfit(fitness_array, average_BDFE_array, 1)
    print(f"Fitted slope for Average BDFE (m_BDFE): {m_BDFE:.4f}")
    print(f"Fitted intercept for Average BDFE (b_BDFE): {b_BDFE:.4f}")

    # Fit for Max BDFE
    m_maxBDFE, b_maxBDFE = np.polyfit(fitness_array, max_BDFE_array, 1)
    print(f"Fitted slope for Max BDFE (m_maxBDFE): {m_maxBDFE:.4f}")
    print(f"Fitted intercept for Max BDFE (b_maxBDFE): {b_maxBDFE:.4f}")

    # Fit for Zero Bin Count using linregress
    slope_zero_bin, intercept_zero_bin, r_value, p_value, std_err = linregress(fitness_array, zero_bin_counts_array)
    print(f"Fitted slope for Zero Bin Count (slope_zero_bin): {slope_zero_bin:.4f}")
    print(f"Fitted intercept for Zero Bin Count (intercept_zero_bin): {intercept_zero_bin:.4f}")

    # Generate fit line values
    fit_fitness = np.linspace(fitness_array.min(), fitness_array.max(), 500)
    fit_DFE = m_DFE * fit_fitness + b_DFE
    fit_BDFE = m_BDFE * fit_fitness + b_BDFE
    fit_maxBDFE = m_maxBDFE * fit_fitness + b_maxBDFE
    fit_zero_bin = slope_zero_bin * fit_fitness + intercept_zero_bin

    # Plotting Metrics vs Fitness
    plt.figure(figsize=(14, 10))

    # Plot Average DFE
    plt.scatter(fitness_array, average_DFE_array, color='blue', label='Average DFE', alpha=0.6)
    plt.plot(fit_fitness, fit_DFE, color='blue', linestyle='--',
             label=f'Fit Average DFE: $<\\Delta_i(t)>$ = {m_DFE*N:.4f} * F(t) + {b_DFE:.4f}')

    # Plot Average BDFE
    plt.scatter(fitness_array, average_BDFE_array, color='green', label='Average BDFE', alpha=0.6)
    plt.plot(fit_fitness, fit_BDFE, color='green', linestyle='--',
             label=f'Fit Average BDFE: $<B\\Delta_i(t)>$ = {m_BDFE*N:.4f} * F(t) + {b_BDFE:.4f}')


    plt.xlabel('Fitness F(t)', fontsize=14)
    plt.ylabel('Metrics', fontsize=14)
    plt.title('SK Model Relaxation: DFE, BDFE, Max BDFE, and Fitness vs Fitness', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plotting Zero Bin Count vs Fitness on a Separate Graph
    plt.figure(figsize=(14, 8))

    norm = zero_bin_counts_array[0]
    # Scatter plot for Zero Bin Count
    plt.scatter(fitness_array, zero_bin_counts_array / norm, color='purple', label='Zero Bin Count', alpha=0.6)
    plt.plot(fit_fitness, fit_zero_bin / norm, color='purple', linestyle='--',
             label=f'Fit Zero Bin Count: Z(t) = {slope_zero_bin*N:.4f} * F(t) + {intercept_zero_bin:.4f}')



    norm = max_BDFE_array[0]
    # Plot Max BDFE
    plt.scatter(fitness_array, max_BDFE_array / norm, color='red', label='Max BDFE', alpha=0.6)
    plt.plot(fit_fitness, fit_maxBDFE / norm, color='red', linestyle='--',
             label=f'Fit Max BDFE: max($B\\Delta_i(t)$) = {m_maxBDFE*N:.4f} * F(t) + {b_maxBDFE:.4f}')

    plt.xlabel('Fitness F(t)', fontsize=14)
    plt.ylabel('Zero Bin Count', fontsize=14)
    plt.title('SK Model Relaxation: Zero Bin Count vs Fitness', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
