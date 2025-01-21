import os
import numpy as np
import matplotlib.pyplot as plt
from code.cmn.cmn_sk import (
    init_h,
    init_J,
    relax_sk,
    compute_fit_slow,
    compute_fit_off
)
from code.cmn.cmn import (init_sigma, curate_sigma_list)

def main():
    """
    Main function to simulate the Sherrington-Kirkpatrick model, perform relaxation,
    compute fitness over flip intervals, and plot the results.
    """
    # Parameters
    random_state = 42  # Seed for reproducibility
    beta = 1.0  # Inverse temperature
    rho = 1.0  # Sparsity of the coupling matrix
    N_values = np.linspace(start=1000, stop=1500, num=5, dtype=int)  # Different values of N
    num_points = 100
    max_fitnesses = []

    # Create directory for saving plots
    output_dir = '../../plots/fit_trajs'
    os.makedirs(output_dir, exist_ok=True)

    # Plot trajectories for different N values
    plt.figure(figsize=(10, 6))
    for N in N_values:
        # Initialize the model
        sigma = init_sigma(N)
        h = init_h(N, random_state=random_state, beta=beta)
        J = init_J(N, random_state=random_state, beta=beta, rho=rho)
        F_off = compute_fit_off(sigma, h, J)  # Compute the offset

        # Perform relaxation
        flip_seq = relax_sk(sigma, h, J)
        num_flips = len(flip_seq)
        ts = list(np.linspace(0, num_flips, num_points, dtype=int))
        sigmas = curate_sigma_list(sigma, flip_seq, ts)

        # Compute fitness over flips
        F_t = [compute_fit_slow(sigma, h, J, F_off) for sigma in sigmas if sigma is not None]
        F_t = F_t[::-1]  # Reverse the list
        max_fitness = F_t[-1]
        max_fitnesses.append(max_fitness)

        # Plot fitness trajectory
        plt.plot(ts, F_t, label=f'N={N}')
        plt.scatter(ts, F_t, s=5)

    plt.xlabel('Number of Flips')
    plt.ylabel('Fitness')
    plt.title('Fitness as a Function of Flips for Different N Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fitness_vs_flips.png'))
    plt.show()  # Display the plot
    plt.close()

    # Plot maximum fitness as a function of N
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, max_fitnesses, 'o-', label='Max Fitness')

    # Perform linear fit
    m, b = np.polyfit(N_values, max_fitnesses, 1)
    fit_line = m * N_values + b
    plt.plot(N_values, fit_line, 'r--', label=f'Fit: y = {m:.4f}x + {b:.4f}')

    plt.xlabel('N')
    plt.ylabel('Maximum Fitness')
    plt.title('Maximum Fitness as a Function of N')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'max_fitness_vs_N.png'))
    plt.show()  # Display the plot

if __name__ == "__main__":
    main()