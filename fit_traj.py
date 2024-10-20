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

def custom_fit_function(t, a, b, c):
    return c + b * t * np.exp(-a * t)

def main():
    """
    Main function to simulate the Sherrington-Kirkpatrick model, perform relaxation,
    compute fitness over flip intervals, and fit the fitness data to a custom model.
    """
    # Parameters
    N = 1000  # Number of spins
    random_state = 42  # Seed for reproducibility
    beta = 1.0  # Inverse temperature
    rho = 1.0  # Sparsity of the coupling matrix

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
        print("Warning: Some flip intervals were not reached during relaxation.")

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

    # Ensure there are enough data points for fitting
    if len(flips_reached) < 2:
        print("Not enough data points for fitting.")
        return

    # Fit to custom function
    try:
        popt_custom, pcov_custom = curve_fit(
            custom_fit_function,
            flips_reached,
            fitnesses,
            p0=[1/N, 1, N]
        )
        perr_custom = np.sqrt(np.diag(pcov_custom))
    except RuntimeError as e:
        print("Custom curve fitting failed:", e)
        popt_custom = [np.nan, np.nan, np.nan]
        perr_custom = [np.nan, np.nan, np.nan]

    # Fit quality check
    if not np.isnan(popt_custom).any():
        print(f"Custom fit parameters:\n a = {popt_custom[0]:.4f} ± {perr_custom[0]:.4f}\n b = {popt_custom[1]:.2f} ± {perr_custom[1]:.2f}\n c = {popt_custom[2]:.2f} ± {perr_custom[2]:.2f}")
    else:
        print("Custom fit parameters are not available due to fitting failure.")

    # Plot fitness as a function of flips
    plt.figure(figsize=(10, 6))
    plt.plot(flips_reached, fitnesses, marker='o', linestyle='-', color='b', label='Data')
    if not np.isnan(popt_custom).any():
        plt.plot(flips_reached, custom_fit_function(flips_reached, *popt_custom), color='r',
                 label=f'Custom Fit: a={popt_custom[0]:.4f}, b={popt_custom[1]:.2f}, c={popt_custom[2]:.2f}')
    plt.xlabel('Number of Flips')
    plt.ylabel('Fitness')
    plt.title('Fitness as a Function of Flips')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(fitnesses)

if __name__ == "__main__":
    main()