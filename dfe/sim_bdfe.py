import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import special

# Import the Funcs module
from misc import Funcs as Fs


def linear_exp_dist(x, _lambda):
    """
    Linear exponential distribution for fitting: -_lambda * x
    """
    return np.log(_lambda / np.sqrt(2)) - _lambda * x

def log_airy_func(x, a, b):
    """
    Log of the Airy function for fitting: log(a * Ai(b * x))
    """
    ai = special.airy(b * x)[0]
    return a + np.log(ai)

def calculate_chi_squared_log(observed_density, expected_log_density, bin_width):
    """
    Calculate the chi-squared statistic for the observed and expected log distributions.
    """
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
    bins = 50         # Number of bins for histograms

    # Initialize spin configuration
    alpha_initial = Fs.init_alpha(N)

    # Initialize external fields
    h = Fs.init_h(N, beta=beta)

    # Initialize coupling matrix with sparsity
    J = Fs.init_J(N, beta=beta, rho=rho)

    # Define desired ranks (sorted in descending order)
    ranks = np.linspace(0, N/2, 11, dtype=int)
    ranks = sorted(ranks, reverse=True)

    # Relax the system using sswm_flip (sswm=True)
    final_alpha, saved_alphas = Fs.relax_sk_ranks(alpha_initial.copy(), h, J, ranks, sswm=True)

    # Create directory for saving histograms
    output_dir = "../Plots/sim_bdfes"
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
                b_airy = np.nan
            else:
                par = 2 * fitness / N
                lam_fitness = np.sqrt(1 / par)
                b_airy = par ** (1/3)

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
            valid = hist > 0
            bin_centers_valid = bin_centers[valid]
            log_hist = np.log(hist[valid])

            initial_guess = [lam_fitness]  # Initial guess for _lambda
            try:
                popt2, pcov2 = curve_fit(linear_exp_dist, bin_centers_valid, log_hist, p0=initial_guess)
                _lambda_fit = popt2[0]
                _lambda_fit_error = np.sqrt(np.diag(pcov2))[0]

                # Generate fitted y data in log-space
                x_fit = np.linspace(0, BDFE.max(), 1000)
                y_fit_log = linear_exp_dist(x_fit, _lambda_fit)
                y_fit = np.exp(y_fit_log)

                # Plot the fitted linear function on log scale
                plt.plot(x_fit, y_fit, color='purple', linestyle='--', label='Fitted Linear')

                # Calculate expected densities for chi-squared
                expected_log_density = linear_exp_dist(bin_centers_valid, _lambda_fit)
                chi_sq = calculate_chi_squared_log(hist[valid], expected_log_density, bin_width)

            except RuntimeError as e:
                print(f"Curve fitting failed for rank {rank} (Linear Fit): {e}")
                _lambda_fit, chi_sq = np.nan, np.nan

            # Fit: Log of Airy function
            try:
                popt_airy, pcov_airy = curve_fit(log_airy_func, bin_centers_valid, log_hist, p0=[1.0, 1.0])
                a_fit_airy, b_fit_airy = popt_airy
                a_fit_airy_error, b_fit_airy_error = np.sqrt(np.diag(pcov_airy))

                # Generate fitted y data for log of Airy function
                y_fit_airy_log = log_airy_func(x_fit, a_fit_airy, b_fit_airy)
                y_fit_airy = np.exp(y_fit_airy_log)

                # Plot the fitted log of Airy function
                plt.plot(x_fit, y_fit_airy, color='blue', linestyle='--', label='Fitted Airy')

            except RuntimeError as e:
                print(f"Curve fitting failed for rank {rank} (Airy Fit): {e}")
                a_fit_airy, b_fit_airy = np.nan, np.nan

            # Annotate the plot with fit parameters and chi-squared values
            annotation_text = ""
            if not np.isnan(_lambda_fit):
                annotation_text += f"Fitted $\\lambda$ = {_lambda_fit:.4f} ± {_lambda_fit_error:.4f}\n"
                annotation_text += f"$\\chi^2$ = {chi_sq:.2f}"
            else:
                annotation_text += "Fitted linear function: Failed"

            if not np.isnan(a_fit_airy):
                annotation_text += f"\nFitted $a_A$ = {a_fit_airy:.4f} ± {a_fit_airy_error:.4f}"
                annotation_text += f"\nFitted $b_A$ = {b_fit_airy:.4f} ± {b_fit_airy_error:.4f}"
            else:
                annotation_text += "\nFitted Airy function: Failed"

            if not np.isnan(lam_fitness):
                annotation_text += f"\nFitness = {fitness:.4f}\nTheoretical $\\lambda = \\sqrt{{N/F(t)}}$ = {lam_fitness:.4f}"
                annotation_text += f"\nTheoretical $b_A = \\left(\\frac{{2 F(t)}}{{N}}\\right)^{{1 / 3}}$ = {b_airy:.4f}"
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
            plot_filename = f'bdfes_rank_{int(rank)}.png'
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, plot_filename))
            plt.close()

            print(f"Histogram for rank {rank} saved as {plot_filename}")
        else:
            print(f"No alpha saved for rank {rank}; skipping histogram.")

    print(f"\nAll histograms have been saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    main()