import os
import numpy as np
import matplotlib.pyplot as plt
from Funcs import (
    init_alpha,
    init_h,
    init_J,
    relax_SK,
    compute_fit_slow,
    calc_rank,
    calc_F_off
)

def main():
    """
    Main function to simulate the Sherrington-Kirkpatrick model, perform relaxation,
    compute fitness over flip intervals, and plot the results.
    """
    # Parameters
    random_state = 42  # Seed for reproducibility
    beta = 1.0  # Inverse temperature
    rho = 1.0  # Sparsity of the coupling matrix
    N_values = np.linspace(start=1000, stop=5000, num=9, dtype=int)  # Different values of N

    max_fitnesses = []

    # Create directory for saving plots
    output_dir = '../Plots/fit_trajs'
    os.makedirs(output_dir, exist_ok=True)

    # Plot trajectories for different N values
    plt.figure(figsize=(10, 6))
    for N in N_values:
        # Initialize the model
        alpha = init_alpha(N)
        h = init_h(N, random_state=random_state, beta=beta)
        J = init_J(N, random_state=random_state, beta=beta, rho=rho)
        flips = np.linspace(start=0, stop=N*0.75, num=50, dtype=int)  # Flip intervals
        F_off = calc_F_off(alpha, h, J)  # Compute the offset

        # Perform relaxation
        final_alpha, saved_alphas, saved_flips, saved_ranks = relax_SK(
            alpha=alpha.copy(),
            his=h,
            Jijs=J,
            flips=flips,
            sswm=True  # Change to False to use Glauber flip
        )

        # Compute fitness over flips
        F_t = [compute_fit_slow(saved_alpha, h, J, F_off) for saved_alpha in saved_alphas if saved_alpha is not None]
        max_fitness = F_t[-1]
        max_fitnesses.append(max_fitness)

        # Plot fitness trajectory
        plt.plot(saved_flips, F_t, label=f'N={N}')
        plt.scatter(saved_flips, F_t, s=5)

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