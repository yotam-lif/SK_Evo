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
    calc_F_off
)

def custom_fit_function(t, a, b):
    return 1 + b * t * np.exp(-a * t)

def main():
    """
    Main function to simulate the Sherrington-Kirkpatrick model, perform relaxation,
    compute fitness over flip intervals, and fit the fitness data to a custom model.
    """
    # Parameters
    random_state = 42  # Seed for reproducibility
    beta = 1.0  # Inverse temperature
    rho = 1.0  # Sparsity of the coupling matrix
    N_values = np.linspace(start=1000, stop=5000, num=9, dtype=int)  # Different values of N
    N_fit = 2000  # Specific value of N for fitting

    final_fitnesses = []
    a_values = []

    # Create directory for saving plots
    output_dir = 'fit_trajs'
    os.makedirs(output_dir, exist_ok=True)

    # Plot trajectories for different N values without fitting
    plt.figure(figsize=(10, 6))
    for N in N_values:
        # Initialize the model
        alpha = init_alpha(N)
        h = init_h(N, random_state=random_state, beta=beta)
        J = init_J(N, random_state=random_state, beta=beta, rho=rho)

        # Calculate F_off
        F_off = calc_F_off(alpha, h, J)

        # Define flip intervals using NumPy (including flip=0)
        num_intervals = 30
        max_flips = int(N * 0.65)  # Example value, adjust as needed
        flip_intervals = np.linspace(1, max_flips, num_intervals + 1, dtype=int)

        # Perform relaxation
        final_alpha, saved_alphas, saved_flips, saved_ranks = relax_SK(
            alpha=alpha.copy(),
            his=h,
            Jijs=J,
            ranks=None,  # No specific ranks to save
            flips=flip_intervals,  # Save at specific flip intervals
            sswm=True  # Change to False to use Glauber flip
        )

        # Check if all flip intervals were reached
        if len(saved_alphas) < len(flip_intervals):
            print(f"Warning: Some flip intervals were not reached during relaxation for N={N}.")

        # Initialize lists to store fitness and flips
        fitnesses = []
        flips_reached = []

        # Iterate over saved alphas and compute fitness
        for flip, saved_alpha in zip(saved_flips, saved_alphas):
            if saved_alpha is not None:
                fitness = compute_fit_slow(saved_alpha, h, J, F_off=F_off)
                fitnesses.append(fitness)
                flips_reached.append(flip)

        # Convert lists to numpy arrays
        fitnesses = np.array(fitnesses)
        flips_reached = np.array(flips_reached)

        # Filter out zero flips to avoid division by zero in the fit function
        non_zero_indices = flips_reached > 0
        fitnesses = fitnesses[non_zero_indices]
        flips_reached = flips_reached[non_zero_indices]

        # Plot fitness as a function of flips
        plt.plot(flips_reached, fitnesses, marker='o', linestyle='-', label=f'N={N}')

        # Store the final fitness for linear fit
        if len(fitnesses) > 0:
            final_fitnesses.append(fitnesses[-1])

        # Fit to custom function
        try:
            popt_custom, _ = curve_fit(
                custom_fit_function,
                flips_reached,
                fitnesses,
                p0=[1/N, 1]
            )
            a_values.append(popt_custom[0])
        except RuntimeError as e:
            print(f"Custom curve fitting failed for N={N}:", e)
            a_values.append(np.nan)

    plt.xlabel('Number of Flips')
    plt.ylabel('Fitness')
    plt.title('Fitness as a Function of Flips for Different N Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fitness_vs_flips.png'))
    plt.close()

    # Plot a as a function of N
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, a_values, 'o-', label='Fitted a values')
    plt.xlabel('N')
    plt.ylabel('Fitted a')
    plt.title('Fitted a as a Function of N')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'a_vs_N.png'))
    plt.close()

if __name__ == "__main__":
    main()