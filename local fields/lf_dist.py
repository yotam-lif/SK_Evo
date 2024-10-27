import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Funcs import (
    init_alpha,
    init_h,
    init_J,
    relax_SK,
    compute_fit_slow,
    calc_rank,
    calc_F_off,
    calc_basic_lfs  # Newly imported function
)

def fit_function(t, m, a, b):
    return m * t**a + b

def main():
    # Parameters
    N = 1000  # Number of spins
    random_state = 42  # Seed for reproducibility
    beta = 0.9
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

    # Create directory for saving plots
    output_dir = '../Plots/lf_dist_plots'
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over saved alphas and compute metrics
    for idx, saved_alpha in enumerate(saved_alphas):
        if saved_alpha is not None:
            lf_dist = calc_basic_lfs(saved_alpha, h, J)
            average_abs_lf = np.mean(np.abs(lf_dist)) * np.sqrt(np.pi / 2)
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
                plt.savefig(os.path.join(output_dir, f'lf_dist_rank_{ranks_to_save[idx]}.png'))
                plt.close()
        else:
            continue

    mean_abs_lfs = np.array(mean_abs_lfs)
    var_lfs_list = np.array(var_lfs_list)
    fitnesses = np.array(fitnesses)

    # Fit and plot Mean and Variance of Local Fields vs Fitness
    # popt_mean, _ = curve_fit(fit_function, fitnesses, mean_abs_lfs, p0=[0, 2, 0.5], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    # popt_var, _ = curve_fit(fit_function, fitnesses, var_lfs_list, p0=[0, 2, 0.5], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))

    plt.figure(figsize=(12, 6))
    plt.scatter(fitnesses, mean_abs_lfs, color='g', label='Mean |Local Fields| * sqrt(pi/2)')
    # plt.plot(fitnesses, fit_function(fitnesses, *popt_mean), color='r', linestyle='--', label=f'Fit Mean: m={popt_mean[0]:.4f}, a={popt_mean[1]:.2f}, b={popt_mean[2]:.2f}')
    plt.scatter(fitnesses, var_lfs_list, color='m', label='Variance of Local Fields')
    # plt.plot(fitnesses, fit_function(fitnesses, *popt_var), color='b', linestyle='--', label=f'Fit Variance: m={popt_var[0]:.4f}, a={popt_var[1]:.2f}, b={popt_var[2]:.2f}')
    plt.xlabel('Fitness')
    plt.ylabel('Mean |Local Fields| and Variance of Local Fields')
    plt.title('Mean |Local Fields| and Variance of Local Fields vs Fitness')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_variance_vs_fitness.png'))
    plt.close()

    # Compute the new function values
    new_function_values = N / 2 - fitnesses / (2 * mean_abs_lfs)

    # Plot the new graph
    plt.figure(figsize=(12, 6))
    plt.plot(fitnesses, new_function_values, label=r'$\frac{N}{2} - \frac{F(t)}{2 \cdot \text{mean\_abs\_lf}(t)}$', color='b')
    plt.xlabel('F(t)')
    plt.ylabel(r'$\frac{N}{2} - \frac{F(t)}{2 \cdot \text{mean\_abs\_lf}(t)}$')
    plt.title(r'$\frac{N}{2} - \frac{F(t)}{2 \cdot \text{mean\_abs\_lf}(t)}$ vs Number of Flips')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'new_function_vs_flips.png'))
    plt.close()

if __name__ == "__main__":
    main()