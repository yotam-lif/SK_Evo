import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit  # Import curve_fit

# Import the Funcs module
from code.cmn import cmn_sk as Fs


def parameterized_function(x, a):
    """
    Parameterized function for fitting: f(x) = a * x * exp(-0.5 * a * x^2)

    Parameters:
    - x (float or array-like): Independent variable.
    - a (float): Parameter to fit.

    Returns:
    - float or array-like: Computed function values.
    """
    return a * -x * np.exp(-0.5 * a * x ** 2)


def main():
    # -------------------------------
    # 1. Initialize SK Environment
    # -------------------------------

    # Parameters
    N = 3000  # Number of spins
    beta = 0.25  # Epistasis strength
    rho = 0.05  # Fraction of non-zero coupling elements
    bins = 60

    # Initialize spin configuration
    alpha_initial = Fs.init_alpha(N)

    # Initialize external fields
    h = Fs.init_h(N, beta=beta)

    # Initialize coupling matrix with sparsity
    J = Fs.init_J(N, beta=beta, rho=rho)

    # Define desired ranks
    ranks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    ranks = sorted(ranks, reverse=True)  # [1000, 900, ..., 0]

    # Relax the system using sswm_flip (sswm=True)
    final_alpha, saved_alphas, flips = Fs.relax_SK(alpha_initial.copy(), h, J, ranks, sswm=True)

    # Debugging: Check which alphas were saved
    print("\n--- Saved Alphas Check ---")
    for rank, alpha in zip(ranks, saved_alphas):
        if alpha is not None:
            print(f"Rank {rank}: Alpha saved.")
        else:
            print(f"Rank {rank}: Alpha NOT saved.")
    print("--------------------------\n")

    # Create directory for saving histograms
    output_dir = "sim_dfes"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over ranks and corresponding saved alphas
    for rank, alpha in zip(ranks, saved_alphas):
        if alpha is not None:
            # Calculate dfe
            DFE = Fs.compute_dfe(alpha, h, J)

            # Set up the plot
            plt.figure(figsize=(8, 6))

            # Compute histogram data
            hist, bin_edges = np.histogram(DFE, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Plot the histogram
            plt.hist(DFE, bins=bins, color='skyblue', edgecolor='black', density=True, alpha=0.6, label='dfe Histogram')

            # If rank is 0, perform fitting and overlay the fitted function
            if rank == 0:
                # Initial guess for parameter 'a'
                initial_guess = [1.0]

                try:
                    # Perform curve fitting
                    popt, pcov = curve_fit(parameterized_function, bin_centers, hist, p0=initial_guess)

                    # Extract the optimal parameter
                    a_fit = popt[0]
                    a_fit_error = np.sqrt(np.diag(pcov))[0]

                    print(f"Fitted parameter a: {a_fit:.4f} ± {a_fit_error:.4f}")

                    # Generate fitted y data
                    x_fit = np.linspace(DFE.min(), DFE.max(), 1000)
                    y_fit = parameterized_function(x_fit, a_fit)

                    # Plot the fitted function
                    plt.plot(x_fit, y_fit, color='red', label=r'$f(x) = a \cdot x \cdot e^{-0.5 \cdot a \cdot x^2}$')
                    plt.legend()

                except RuntimeError as e:
                    print(f"Curve fitting failed for rank {rank}: {e}")

            # Title and labels with parameters
            plt.title(f'dfe Histogram at Rank {rank}; N={N}, β={beta}, ρ={rho}')
            plt.xlabel('Fitness Effect')
            plt.ylabel('Density')

            # Save the plot
            plot_filename = f'dfe_rank_{rank}.png'
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, plot_filename))
            plt.close()

            print(f"Histogram for rank {rank} saved as {plot_filename}")
        else:
            print(f"No alpha saved for rank {rank}; skipping histogram.")

    print(f"\nAll histograms have been saved in the '{output_dir}' directory.")


if __name__ == "__main__":
    main()
