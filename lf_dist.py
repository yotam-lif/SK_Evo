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
    N = 2000  # Number of spins
    random_state = 42  # Seed for reproducibility
    beta = 1.0  # Inverse temperature
    rho = 1.0  # Sparsity of the coupling matrix

    # Initialize the model
    alpha = init_alpha(N)
    h = init_h(N, random_state=random_state, beta=beta)
    J = init_J(N, random_state=random_state, beta=beta, rho=rho)
    initial_fitness = compute_fit_slow(alpha, h, J)
    print(f'Initial fitness = {initial_fitness:.4f}')

    # Uncomment the following line if F_off calculation is needed
    F_off = calc_F_off(alpha, h, J)

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
    mean_abs_lfs = []
    var_lfs_list = []
    flip_counts = []
    fitnesses = []

    # Iterate over saved alphas and compute metrics
    for idx, saved_alpha in enumerate(saved_alphas):
        if saved_alpha is not None:
            # Calculate local fields
            lf_dist = calc_basic_lfs(saved_alpha, h, J)

            # Compute average of absolute local fields
            average_abs_lf = np.mean(np.abs(lf_dist))
            mean_abs_lfs.append(average_abs_lf)

            # Compute variance of local fields
            var_lfs = np.var(lf_dist)
            var_lfs_list.append(var_lfs)

            # Record the number of flips at this rank
            flip_counts.append(saved_flips[idx])

            # Compute fitness for the saved configuration
            fitness = compute_fit_slow(saved_alpha, h, J)
            fitnesses.append(fitness)
        else:
            # If no configuration was saved at this rank, skip
            print(f"No configuration saved at rank index {idx}.")

    # Convert lists to numpy arrays for easier handling
    mean_abs_lfs = np.array(mean_abs_lfs)
    var_lfs_list = np.array(var_lfs_list)
    flip_counts = np.array(flip_counts)
    fitnesses = np.array(fitnesses)

    # Plotting Mean and Variance vs Number of Flips
    plt.figure(figsize=(12, 5))

    # Mean of |Local Fields| vs Number of Flips
    plt.subplot(1, 2, 1)
    plt.plot(flip_counts, mean_abs_lfs, marker='o', linestyle='-',
             color='b', label='Mean |Local Fields|')
    plt.xlabel('Number of Flips')
    plt.ylabel('Mean |Local Fields|')
    plt.title('Mean |Local Fields| vs Number of Flips')
    plt.legend()
    plt.grid(True)

    # Variance of Local Fields vs Number of Flips
    plt.subplot(1, 2, 2)
    plt.plot(flip_counts, var_lfs_list, marker='s', linestyle='--',
             color='r', label='Variance of Local Fields')
    plt.xlabel('Number of Flips')
    plt.ylabel('Variance of Local Fields')
    plt.title('Variance of Local Fields vs Number of Flips')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plotting Mean and Variance vs Fitness
    plt.figure(figsize=(12, 5))

    # Mean of |Local Fields| vs Fitness
    plt.subplot(1, 2, 1)
    plt.scatter(fitnesses, mean_abs_lfs, color='g', label='Mean |Local Fields|')
    plt.xlabel('Fitness')
    plt.ylabel('Mean |Local Fields|')
    plt.title('Mean |Local Fields| vs Fitness')
    plt.legend()
    plt.grid(True)

    # Variance of Local Fields vs Fitness
    plt.subplot(1, 2, 2)
    plt.scatter(fitnesses, var_lfs_list, color='m', label='Variance of Local Fields')
    plt.xlabel('Fitness')
    plt.ylabel('Variance of Local Fields')
    plt.title('Variance of Local Fields vs Fitness')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()