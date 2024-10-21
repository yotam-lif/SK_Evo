import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Import the Funcs module
import Funcs as Fs


def linear_func(x, m, a):
    """
    Linear function for fitting: f(x) = m * x + a

    Parameters:
    - x (float or array-like): Independent variable.
    - m (float): Slope of the line.
    - a (float): Intercept of the line.

    Returns:
    - float or array-like: Computed function values.
    """
    return m * x + a


def calculate_chi_squared_log(observed_density, expected_log_density, bin_width):
    """
    Calculate the chi-squared statistic for the observed and expected log distributions.

    Parameters:
    - observed_density (array-like): Observed density values from the histogram.
    - expected_log_density (array-like): Expected log density values from the model.
    - bin_width (float): Width of the histogram bins.

    Returns:
    - float: Chi-squared value.
    """
    # To avoid division by zero, add a small epsilon to observed_density
    epsilon = 1e-10
    observed_density = np.where(observed_density == 0, epsilon, observed_density)
    log_observed = np.log(observed_density)
    chi_sq = np.sum(((log_observed - expected_log_density) ** 2) / expected_log_density) * bin_width
    return chi_sq


def main():
    # -------------------------------
    # 1. Initialize SK Environment
    # -------------------------------

    # Parameters
    N = 2000          # Number of spins
    beta = 1.0       # Epistasis strength (inverse temperature)
    rho = 1.0        # Fraction of non-zero coupling elements
    bins = 40         # Number of bins for histograms

    # Initialize spin configuration
    alpha_initial = Fs.init_alpha(N)

    # Initialize external fields
    h = Fs.init_h(N, beta=beta)

    # Initialize coupling matrix with sparsity
    J = Fs.init_J(N, beta=beta, rho=rho)

    # Define desired ranks (sorted in descending order)
    ranks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    ranks = np.array(ranks)
    ranks *= 1
    ranks = sorted(ranks, reverse=True)  # [1000, 900, ..., 0]

    # Relax the system using sswm_flip (sswm=True)
    final_alpha, saved_alphas, saved_flips, saved_ranks = Fs.relax_SK(alpha_initial.copy(), h, J, ranks, sswm=True)

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
                lam_fitness = np.sqrt(N / (2*fitness))

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

            # Set the y-axis to logarithmic scale
            plt.yscale('log')

            # Fit: Linear fit to log(density)
            # To avoid taking log of zero, ensure hist > 0
            valid = hist > 0
            bin_centers_valid = bin_centers[valid]
            log_hist = np.log(hist[valid])

            initial_guess = [ -1.0, np.log(np.max(hist)) ]  # Initial guesses for m and a
            try:
                popt2, pcov2 = curve_fit(linear_func, bin_centers_valid, log_hist, p0=initial_guess)
                m_fit, a_fit = popt2
                m_fit_error, a_fit_error = np.sqrt(np.diag(pcov2))

                # Generate fitted y data in log-space
                x_fit = np.linspace(0, BDFE.max(), 1000)
                y_fit_log = linear_func(x_fit, m_fit, a_fit)
                y_fit = np.exp(y_fit_log)

                # Plot the fitted linear function on log scale
                plt.plot(x_fit, y_fit, color='purple', linestyle='--',
                         label=f'Fitted Linear (log scale)\n$y = {m_fit:.4f}x + {a_fit:.4f}$')

                # Calculate expected densities for chi-squared
                expected_log_density = linear_func(bin_centers_valid, m_fit, a_fit)
                chi_sq = calculate_chi_squared_log(hist[valid], expected_log_density, bin_width)

            except RuntimeError as e:
                print(f"Curve fitting failed for rank {rank} (Linear Fit): {e}")
                m_fit, a_fit, chi_sq = np.nan, np.nan, np.nan

            # Annotate the plot with fit parameters and chi-squared values
            annotation_text = ""
            if not np.isnan(m_fit):
                annotation_text += f"Fitted $m$ = {m_fit:.4f} ± {m_fit_error:.4f}\n"
                annotation_text += f"Fitted $a$ = {a_fit:.4f} ± {a_fit_error:.4f}\n"
                annotation_text += f"$\\chi^2$ = {chi_sq:.2f}"
            else:
                annotation_text += "Fitted linear function: Failed"

            if not np.isnan(lam_fitness):
                annotation_text += f"\nFitness = {fitness:.4f}\nTheoretical $\\lambda = \\sqrt{{N/F(t)}}$ = {lam_fitness:.4f}"
            else:
                annotation_text += "\nFitness: Invalid (<= 0)"

            # Position the annotation
            plt.text(0.95, 0.95, annotation_text, transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

            # Title and labels with parameters
            plt.title(f'BDFE Histogram at Rank {rank}; N={N}, $\\beta$={beta}, $\\rho$={rho}', fontsize=14)
            plt.xlabel('Beneficial Fitness Effect', fontsize=12)
            plt.ylabel('Density (log scale)', fontsize=12)  # Updated ylabel

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
