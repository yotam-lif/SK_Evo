import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Import the Funcs module
import Funcs as Fs


def exponential_distribution(x, lam):
    """
    Exponential distribution function: f(x) = lam * exp(-lam * x)

    Parameters:
    - x (float or array-like): Independent variable.
    - lam (float): Rate parameter of the exponential distribution.

    Returns:
    - float or array-like: Computed function values.
    """
    return lam * np.exp(-lam * x)


def calculate_chi_squared(observed_density, expected_density, bin_width):
    """
    Calculate the chi-squared statistic for the observed and expected distributions.

    Parameters:
    - observed_density (array-like): Observed density values from the histogram.
    - expected_density (array-like): Expected density values from the model.
    - bin_width (float): Width of the histogram bins.

    Returns:
    - float: Chi-squared value.
    """
    # To avoid division by zero, add a small epsilon to expected
    epsilon = 1e-10
    expected_density = np.where(expected_density == 0, epsilon, expected_density)
    chi_sq = np.sum(((observed_density - expected_density) ** 2) / expected_density) * bin_width
    return chi_sq


def main():
    # -------------------------------
    # 1. Initialize SK Environment
    # -------------------------------

    # Parameters
    N = 3000          # Number of spins
    beta = 0.25       # Epistasis strength (inverse temperature)
    rho = 0.05        # Fraction of non-zero coupling elements
    bins = 60         # Number of bins for histograms

    # Initialize spin configuration
    alpha_initial = Fs.init_alpha(N)

    # Initialize external fields
    h = Fs.init_h(N, beta=beta)

    # Initialize coupling matrix with sparsity
    J = Fs.init_J(N, beta=beta, rho=rho)

    # Define desired ranks (sorted in descending order)
    ranks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    ranks = sorted(ranks, reverse=True)  # [1000, 900, ..., 0]

    # Relax the system using sswm_flip (sswm=True)
    final_alpha, saved_alphas, flips = Fs.relax_SK(alpha_initial.copy(), h, J, ranks, sswm=True)

    # Create directory for saving histograms
    output_dir = "sim_bdfes"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over ranks and corresponding saved alphas
    for rank, alpha in zip(ranks, saved_alphas):
        if alpha is not None:
            # Calculate BDFE
            BDFE, bdfe_indices = Fs.calc_BDFE(alpha, h, J)

            # Check if there are any beneficial effects
            if len(BDFE) == 0:
                print(f"No beneficial effects at rank {rank}; skipping histogram.")
                continue

            # Calculate fitness at this rank using compute_fit_slow
            fitness = Fs.compute_fit_slow(alpha, h, J)

            # Ensure fitness is positive to avoid division by zero
            if fitness <= 0:
                print(f"Fitness is non-positive ({fitness}) at rank {rank}; cannot compute lambda = 1/fitness.")
                lam_fitness = np.nan
            else:
                lam_fitness = N / fitness

            # Set up the plot
            plt.figure(figsize=(10, 7))
            sns.set(style="whitegrid")  # Apply seaborn style for better aesthetics

            # Compute histogram data
            hist, bin_edges = np.histogram(BDFE, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]

            # Plot the histogram
            plt.hist(BDFE, bins=bins, color='lightgreen', edgecolor='black',
                     density=True, alpha=0.6, label='BDFE Histogram')

            # Fit 1: Exponential distribution with free lambda
            initial_guess = [1.0]
            try:
                popt1, pcov1 = curve_fit(exponential_distribution, bin_centers, hist, p0=initial_guess, bounds=(0, np.inf))
                lam_fit = popt1[0]
                lam_fit_error = np.sqrt(np.diag(pcov1))[0]

                # Generate fitted y data
                x_fit = np.linspace(0, BDFE.max(), 1000)
                y_fit = exponential_distribution(x_fit, lam_fit)

                # Plot the fitted exponential
                plt.plot(x_fit, y_fit, color='red', linestyle='--',
                         label=f'Fitted Exponential ($\\lambda$ = {lam_fit:.4f})')

                # Calculate expected densities for chi-squared
                expected_density_fit = exponential_distribution(bin_centers, lam_fit)
                chi_sq = calculate_chi_squared(hist, expected_density_fit, bin_width)

            except RuntimeError as e:
                print(f"Curve fitting failed for rank {rank} (Fit 1): {e}")
                lam_fit = np.nan
                chi_sq = np.nan

            # Plot the theoretical exponential with lambda = 1 / fitness
            if not np.isnan(lam_fitness):
                y_fit_fitness = exponential_distribution(x_fit, lam_fitness)
                plt.plot(x_fit, y_fit_fitness, color='blue', linestyle=':',
                         label=f'Theoretical Exponential ($\\lambda$ = N/Fitness = {lam_fitness:.4f})')

            # Annotate the plot with fit parameters and chi-squared values
            annotation_text = ""
            if not np.isnan(lam_fit):
                annotation_text += f"Fitted $\\lambda$ = {lam_fit:.4f} ± {lam_fit_error:.4f}\n$\\chi^2$ = {chi_sq:.2f}"
            else:
                annotation_text += "Fitted $\\lambda$: Failed"

            if not np.isnan(lam_fitness):
                annotation_text += f"\nFitness = {fitness:.4f}\nTheoretical $\\lambda$ = 1/Fitness = {lam_fitness:.4f}"
            else:
                annotation_text += "\nFitness: Invalid (<= 0)"

            # Position the annotation
            plt.text(0.95, 0.95, annotation_text, transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

            # Title and labels with parameters
            plt.title(f'BDFE Histogram at Rank {rank}; N={N}, $\\beta$={beta}, $\\rho$={rho}', fontsize=14)
            plt.xlabel('Beneficial Fitness Effect', fontsize=12)
            plt.ylabel('Density', fontsize=12)

            # Legend
            plt.legend()

            # Save the plot
            plot_filename = f'bdfes_rank_{rank}.png'
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, plot_filename))
            plt.close()

            print(f"Histogram for rank {rank} saved as {plot_filename}")
        else:
            print(f"No alpha saved for rank {rank}; skipping histogram.")

    print(f"\nAll histograms have been saved in the '{output_dir}' directory.")


if __name__ == "__main__":
    main()