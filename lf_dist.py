import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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

def fit_function(t, m, a, b):
    return m * t**a + b

def main():
    # Parameters
    N = 2000  # Number of spins
    random_state = 42  # Seed for reproducibility
    beta = 1.0  # Inverse temperature
    rho = 1.0  # Sparsity of the coupling matrix
    # Define the number of lowest ranks to plot
    n = 8  # Example value, adjust as needed

    # Initialize the model
    alpha = init_alpha(N)
    h = init_h(N, random_state=random_state, beta=beta)
    J = init_J(N, random_state=random_state, beta=beta, rho=rho)
    initial_fitness = compute_fit_slow(alpha, h, J)
    print(f'Initial fitness = {initial_fitness:.4f}')

    # Uncomment the following line if F_off calculation is needed
    F_off = calc_F_off(alpha, h, J)

    # Define ranks at which to save the configurations
    initial_rank = calc_rank(alpha, h, J)
    num_saves = 30
    if initial_rank < num_saves:
        ranks_to_save = list(range(initial_rank, -1, -1))
    else:
        step = initial_rank // num_saves
        ranks_to_save = list(range(initial_rank, -1, -step))
        if 0 not in ranks_to_save:
            ranks_to_save.append(0)

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
    mean_abs_lfs = []
    var_lfs_list = []
    flip_counts = []
    fitnesses = []

    # Subset of ranks to plot local field distributions
    ranks_to_plot = ranks_to_save[-n:]  # Get the n lowest ranks

    # Iterate over saved alphas and compute metrics
    for idx, saved_alpha in enumerate(saved_alphas):
        if saved_alpha is not None:
            lf_dist = calc_basic_lfs(saved_alpha, h, J)
            average_abs_lf = np.mean(np.abs(lf_dist))
            mean_abs_lfs.append(average_abs_lf)
            var_lfs = np.var(lf_dist)
            var_lfs_list.append(var_lfs)
            flip_counts.append(saved_flips[idx])
            fitness = compute_fit_slow(saved_alpha, h, J)
            fitnesses.append(fitness)

            if ranks_to_save[idx] in ranks_to_plot:
                plt.figure()
                plt.hist(lf_dist, bins=50, alpha=0.75, color='blue', edgecolor='black')
                plt.title(f'Local Field Distribution at Rank {ranks_to_save[idx]}')
                plt.xlabel('Local Field')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.show()
        else:
            continue

    mean_abs_lfs = np.array(mean_abs_lfs)
    var_lfs_list = np.array(var_lfs_list)
    flip_counts = np.array(flip_counts)
    fitnesses = np.array(fitnesses)

    # Fit and plot Mean of |Local Fields| vs Fitness
    popt_mean, _ = curve_fit(fit_function, fitnesses, mean_abs_lfs, p0=[0, 2, 1])
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(fitnesses, mean_abs_lfs, color='g', label='Mean |Local Fields|')
    plt.plot(fitnesses, fit_function(fitnesses, *popt_mean), color='r', label=f'Fit: m={popt_mean[0]:.4f}, a={popt_mean[1]:.2f}, b={popt_mean[2]:.2f}')
    plt.xlabel('Fitness')
    plt.ylabel('Mean |Local Fields|')
    plt.title('Mean |Local Fields| vs Fitness')
    plt.legend()
    plt.grid(True)

    # Fit and plot Variance of Local Fields vs Fitness
    popt_var, _ = curve_fit(fit_function, fitnesses, var_lfs_list, p0=[0, 2, 1])
    plt.subplot(1, 2, 2)
    plt.scatter(fitnesses, var_lfs_list, color='m', label='Variance of Local Fields')
    plt.plot(fitnesses, fit_function(fitnesses, *popt_var), color='r', label=f'Fit: m={popt_var[0]:.4f}, a={popt_var[1]:.2f}, b={popt_var[2]:.2f}')
    plt.xlabel('Fitness')
    plt.ylabel('Variance of Local Fields')
    plt.title('Variance of Local Fields vs Fitness')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()