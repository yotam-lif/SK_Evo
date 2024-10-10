import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr  # For calculating correlation coefficients
from scipy.optimize import curve_fit
import pandas as pd  # For saving correlation data

# Import the Funcs module
import Funcs as Fs


def main():
    # -------------------------------
    # 1. Initialize SK Environment
    # -------------------------------

    # Parameters
    N = 2000  # Number of spins
    beta = 1  # Epistasis strength
    rho = 1  # Fraction of non-zero coupling elements
    random_state = 42  # Seed for reproducibility

    # Initialize spin configuration
    alpha_initial = Fs.init_alpha(N)

    # Initialize external fields
    h = Fs.init_h(N, beta=beta, random_state=random_state)

    # Initialize coupling matrix with sparsity
    J = Fs.init_J(N, beta=beta, rho=rho, random_state=random_state)

    # Define desired ranks (example: [0, 100, 200, ..., 1000])
    ranks = list(range(0, 1100, 100))  # [0, 100, 200, ..., 1000]
    ranks = sorted(ranks, reverse=True)  # Sort in descending order

    # Relax the system using sswm_flip (sswm=True)
    # Now returns flips (number of mutations up to each rank)
    final_alpha, saved_alphas, flips = Fs.relax_SK(alpha_initial.copy(), h, J, ranks, sswm=True)

    # Calculate initial kis values
    ki_initial = Fs.calc_kis(alpha_initial, h, J)

    # Calculate initial basic local fields
    basic_local_fields_initial = Fs.calc_basic_lfs(alpha_initial, h, J)

    # Calculate absolute basic local fields
    abs_basic_local_fields_initial = np.abs(basic_local_fields_initial)

    # Debugging: Check which alphas were saved
    print("\n--- Saved Alphas Check ---")
    for rank, alpha in zip(ranks, saved_alphas):
        if alpha is not None:
            print(f"Rank {rank}: Alpha saved.")
        else:
            print(f"Rank {rank}: Alpha NOT saved.")
    print("--------------------------\n")

    # -------------------------------
    # 2. Setup Output Directories
    # -------------------------------

    # Determine the current script's directory
    curr_dir_path = os.path.dirname(os.path.realpath(__file__))

    # Define main output directory
    main_output_dir = os.path.join(curr_dir_path, "sim_correlations")
    os.makedirs(main_output_dir, exist_ok=True)

    # Define subdirectories for each correlation type and plots
    subdirs = {
        'kis_correlation': 'kis_correlation',
        'local_fields_correlation': 'local_fields_correlation',
        'absolute_local_fields_correlation': 'absolute_local_fields_correlation',
        'variance_plots': 'variance_plots',
        'correlation_summary_plots': 'correlation_summary_plots'
    }

    # Create subdirectories
    for key, subdir in subdirs.items():
        path = os.path.join(main_output_dir, subdir)
        os.makedirs(path, exist_ok=True)

    # -------------------------------
    # 3. Initialize Lists to Store Data
    # -------------------------------

    correlation_kis = []
    correlation_local_fields = []
    correlation_absolute_local_fields = []

    # For variance calculations
    variance_delta_y_kis = []
    variance_delta_y_lf = []
    variance_delta_y_abs_lf = []
    mutation_counts = flips  # This serves as our "time" variable

    # -------------------------------
    # 4. Iterate Over Ranks and Generate Plots
    # -------------------------------

    for idx, (rank, alpha) in enumerate(zip(ranks, saved_alphas)):
        if alpha is not None:
            # -------------------------------
            # 4.1 Calculate Current Values
            # -------------------------------

            # Calculate current kis values
            ki_current = Fs.calc_kis(alpha, h, J)

            # Calculate current basic local fields
            basic_local_fields_current = Fs.calc_basic_lfs(alpha, h, J)

            # Calculate current absolute basic local fields
            abs_basic_local_fields_current = np.abs(basic_local_fields_current)

            # -------------------------------
            # 4.2 Compute Displacements and Variances
            # -------------------------------

            # For kis
            delta_y_kis = ki_current - ki_initial
            var_delta_y_kis = np.var(delta_y_kis)
            variance_delta_y_kis.append(var_delta_y_kis)

            # For basic local fields
            delta_y_lf = basic_local_fields_current - basic_local_fields_initial
            var_delta_y_lf = np.var(delta_y_lf)
            variance_delta_y_lf.append(var_delta_y_lf)

            # For absolute basic local fields
            delta_y_abs_lf = abs_basic_local_fields_current - abs_basic_local_fields_initial
            var_delta_y_abs_lf = np.var(delta_y_abs_lf)
            variance_delta_y_abs_lf.append(var_delta_y_abs_lf)

            # -------------------------------
            # 4.3 kis Correlation
            # -------------------------------

            # Compute Pearson correlation coefficient for kis
            pearson_r_kis, p_value_kis = pearsonr(ki_initial, ki_current)

            # Compute Spearman correlation coefficient for kis
            spearman_r_kis, p_value_spearman_kis = spearmanr(ki_initial, ki_current)

            # Append kis correlation data
            correlation_kis.append({
                'rank': rank,
                'pearson_r_kis': pearson_r_kis,
                'p_value_kis': p_value_kis,
                'spearman_r_kis': spearman_r_kis,
                'p_value_spearman_kis': p_value_spearman_kis
            })

            # Set up the kis correlation plot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=ki_initial, y=ki_current, alpha=0.5, edgecolor=None, s=20)

            # Plot reference lines
            max_val_kis = max(np.max(ki_initial), np.max(ki_current))
            min_val_kis = min(np.min(ki_initial), np.min(ki_current))
            plt.plot([min_val_kis, max_val_kis], [min_val_kis, max_val_kis],
                     color='red', linestyle='--', label='y = x')
            plt.plot([min_val_kis, max_val_kis], [-min_val_kis, -max_val_kis],
                     color='green', linestyle='--', label='y = -x')

            # Annotate with Pearson and Spearman correlations
            plt.text(0.05, 0.95,
                     f'Pearson r = {pearson_r_kis:.4f}\nSpearman rho = {spearman_r_kis:.4f}',
                     transform=plt.gca().transAxes,
                     fontsize=12,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            # Title and labels
            plt.title(f'kis Correlation at Rank {rank}; N={N}, β={beta}, ρ={rho}')
            plt.xlabel('Initial kis')
            plt.ylabel(f'kis at Rank {rank}')
            plt.legend()
            plt.tight_layout()

            # Save the kis correlation plot
            plot_filename_kis = f'correlation_kis_rank_{rank}.png'
            plt.savefig(os.path.join(main_output_dir, subdirs['kis_correlation'], plot_filename_kis))
            plt.close()

            print(f"kis Correlation scatter plot for rank {rank} saved as {plot_filename_kis}")

            # -------------------------------
            # 4.4 Basic Local Fields Correlation
            # -------------------------------

            # Compute Pearson correlation coefficient for basic local fields
            pearson_r_lf, p_value_lf = pearsonr(basic_local_fields_initial, basic_local_fields_current)

            # Compute Spearman correlation coefficient for basic local fields
            spearman_r_lf, p_value_spearman_lf = spearmanr(basic_local_fields_initial, basic_local_fields_current)

            # Append basic local fields correlation data
            correlation_local_fields.append({
                'rank': rank,
                'pearson_r_local_fields': pearson_r_lf,
                'p_value_local_fields': p_value_lf,
                'spearman_r_local_fields': spearman_r_lf,
                'p_value_spearman_lf': p_value_spearman_lf
            })

            # Set up the basic local fields correlation plot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=basic_local_fields_initial, y=basic_local_fields_current,
                            alpha=0.5, edgecolor=None, s=20)

            # Plot reference line y = x
            max_val_lf = max(np.max(basic_local_fields_initial), np.max(basic_local_fields_current))
            min_val_lf = min(np.min(basic_local_fields_initial), np.min(basic_local_fields_current))
            plt.plot([min_val_lf, max_val_lf], [min_val_lf, max_val_lf],
                     color='red', linestyle='--', label='y = x')

            # Annotate with Pearson and Spearman correlations
            plt.text(0.05, 0.95,
                     f'Pearson r = {pearson_r_lf:.4f}\nSpearman rho = {spearman_r_lf:.4f}',
                     transform=plt.gca().transAxes,
                     fontsize=12,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            # Title and labels
            plt.title(f'Basic Local Fields Correlation at Rank {rank}; N={N}, β={beta}, ρ={rho}')
            plt.xlabel('Initial Basic Local Fields')
            plt.ylabel(f'Basic Local Fields at Rank {rank}')
            plt.legend()
            plt.tight_layout()

            # Save the basic local fields correlation plot
            plot_filename_lf = f'correlation_local_fields_rank_{rank}.png'
            plt.savefig(os.path.join(main_output_dir, subdirs['local_fields_correlation'], plot_filename_lf))
            plt.close()

            print(f"Basic Local Fields Correlation scatter plot for rank {rank} saved as {plot_filename_lf}")

            # -------------------------------
            # 4.5 Absolute Basic Local Fields Correlation
            # -------------------------------

            # Compute Pearson correlation coefficient for absolute basic local fields
            pearson_r_abs_lf, p_value_abs_lf = pearsonr(abs_basic_local_fields_initial,
                                                        abs_basic_local_fields_current)

            # Compute Spearman correlation coefficient for absolute basic local fields
            spearman_r_abs_lf, p_value_spearman_abs_lf = spearmanr(abs_basic_local_fields_initial,
                                                                   abs_basic_local_fields_current)

            # Append absolute basic local fields correlation data
            correlation_absolute_local_fields.append({
                'rank': rank,
                'pearson_r_abs_local_fields': pearson_r_abs_lf,
                'p_value_abs_local_fields': p_value_abs_lf,
                'spearman_r_abs_local_fields': spearman_r_abs_lf,
                'p_value_spearman_abs_lf': p_value_spearman_abs_lf
            })

            # Set up the absolute basic local fields correlation plot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=abs_basic_local_fields_initial, y=abs_basic_local_fields_current,
                            alpha=0.5, edgecolor=None, s=20)

            # Plot reference line y = x
            max_val_abs_lf = max(np.max(abs_basic_local_fields_initial), np.max(abs_basic_local_fields_current))
            min_val_abs_lf = min(np.min(abs_basic_local_fields_initial), np.min(abs_basic_local_fields_current))
            plt.plot([min_val_abs_lf, max_val_abs_lf], [min_val_abs_lf, max_val_abs_lf],
                     color='red', linestyle='--', label='y = x')

            # Annotate with Pearson and Spearman correlations
            plt.text(0.05, 0.95,
                     f'Pearson r = {pearson_r_abs_lf:.4f}\nSpearman rho = {spearman_r_abs_lf:.4f}',
                     transform=plt.gca().transAxes,
                     fontsize=12,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            # Title and labels
            plt.title(f'Absolute Basic Local Fields Correlation at Rank {rank}; N={N}, β={beta}, ρ={rho}')
            plt.xlabel('Initial |Basic Local Fields|')
            plt.ylabel(f'|Basic Local Fields| at Rank {rank}')
            plt.legend()
            plt.tight_layout()

            # Save the absolute basic local fields correlation plot
            plot_filename_abs_lf = f'correlation_absolute_local_fields_rank_{rank}.png'
            plt.savefig(os.path.join(main_output_dir, subdirs['absolute_local_fields_correlation'],
                                     plot_filename_abs_lf))
            plt.close()

            print(f"Absolute Basic Local Fields Correlation scatter plot for rank {rank} saved as {plot_filename_abs_lf}")

    # -------------------------------
    # 5. Save All Correlation Data to CSV
    # -------------------------------

    # After processing all ranks
    if correlation_kis and correlation_local_fields and correlation_absolute_local_fields:
        # Merge all correlation data based on ranks
        df_kis = pd.DataFrame(correlation_kis).set_index('rank')
        df_lf = pd.DataFrame(correlation_local_fields).set_index('rank')
        df_abs_lf = pd.DataFrame(correlation_absolute_local_fields).set_index('rank')

        # Combine all DataFrames
        df_all = df_kis.join(df_lf).join(df_abs_lf)

        # Define the CSV filename
        correlation_filename = 'correlation_coefficients.csv'

        # Save to CSV in the main output directory
        df_all.to_csv(os.path.join(main_output_dir, correlation_filename))
        print(f"\nCorrelation coefficients saved as {correlation_filename} in '{main_output_dir}' directory.")
    else:
        print("\nNo correlation data to save.")

    print(f"\nAll correlation scatter plots have been saved in the '{main_output_dir}' directory.")

    # -------------------------------
    # 6. Plot Variance vs. Time and Perform Linear Fitting
    # -------------------------------

    # Convert lists to numpy arrays
    mutation_counts = np.array(mutation_counts)
    variance_delta_y_kis = np.array(variance_delta_y_kis)
    variance_delta_y_lf = np.array(variance_delta_y_lf)
    variance_delta_y_abs_lf = np.array(variance_delta_y_abs_lf)

    # Function to perform linear fitting and calculate chi-squared
    def linear_fit_and_chi_squared(x, y):
        # Define linear function
        def linear_func(x, m):
            return m * x

        # Perform linear fit
        popt, pcov = curve_fit(linear_func, x, y)
        m = popt[0]

        # Calculate fitted values
        y_fit = linear_func(x, m)

        # Calculate chi-squared
        chi_squared = np.sum(((y - y_fit) ** 2) / y_fit)

        return m, chi_squared, y_fit

    # Perform linear fitting for variance of Δy for kis
    m_kis, chi_squared_kis, y_fit_kis = linear_fit_and_chi_squared(mutation_counts, variance_delta_y_kis)

    # Perform linear fitting for variance of Δy for basic local fields
    m_lf, chi_squared_lf, y_fit_lf = linear_fit_and_chi_squared(mutation_counts, variance_delta_y_lf)

    # Perform linear fitting for variance of Δy for absolute basic local fields
    m_abs_lf, chi_squared_abs_lf, y_fit_abs_lf = linear_fit_and_chi_squared(mutation_counts, variance_delta_y_abs_lf)

    # Plot variance of Δy vs. mutations for kis
    plt.figure(figsize=(8, 6))
    plt.scatter(mutation_counts, variance_delta_y_kis, label='Variance of Δy (kis)', color='blue')
    label = f'm = {m_kis:.4e}, χ² = {chi_squared_kis:.4f}'
    plt.plot(mutation_counts, y_fit_kis, label=label, color='red')
    plt.title('Variance of Δy (kis) vs. Number of Mutations')
    plt.xlabel('Number of Mutations')
    plt.ylabel('Variance of Δy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    variance_kis_plot = 'variance_delta_y_kis_vs_mutations.png'
    plt.savefig(os.path.join(main_output_dir, subdirs['variance_plots'], variance_kis_plot))
    plt.close()

    # Print the fit parameters
    print(f'Variance of Δy (kis) fit: m = {m_kis:.4e}')
    print(f'Chi-squared for Δy (kis) fit: {chi_squared_kis:.4f}')

    # Plot variance of Δy vs. mutations for basic local fields
    plt.figure(figsize=(8, 6))
    plt.scatter(mutation_counts, variance_delta_y_lf, label='Variance of Δy (Basic LF)', color='green')
    label = f'm = {m_lf:.4e}, χ² = {chi_squared_lf:.4f}'
    plt.plot(mutation_counts, y_fit_lf, label=label, color='red')
    plt.title('Variance of Δy (Basic Local Fields) vs. Number of Mutations')
    plt.xlabel('Number of Mutations')
    plt.ylabel('Variance of Δy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    variance_lf_plot = 'variance_delta_y_lf_vs_mutations.png'
    plt.savefig(os.path.join(main_output_dir, subdirs['variance_plots'], variance_lf_plot))
    plt.close()

    # Print the fit parameters
    print(f'Variance of Δy (Basic LF) fit: m = {m_lf:.4e}')
    print(f'Chi-squared for Δy (Basic LF) fit: {chi_squared_lf:.4f}')

    # Plot variance of Δy vs. mutations for absolute basic local fields
    plt.figure(figsize=(8, 6))
    plt.scatter(mutation_counts, variance_delta_y_abs_lf, label='Variance of Δy (Abs Basic LF)', color='purple')
    label = f'm = {m_abs_lf:.4e}, χ² = {chi_squared_abs_lf:.4f}'
    plt.plot(mutation_counts, y_fit_abs_lf, label=label, color='red')
    plt.title('Variance of Δy (Absolute Basic Local Fields) vs. Number of Mutations')
    plt.xlabel('Number of Mutations')
    plt.ylabel('Variance of Δy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    variance_abs_lf_plot = 'variance_delta_y_abs_lf_vs_mutations.png'
    plt.savefig(os.path.join(main_output_dir, subdirs['variance_plots'], variance_abs_lf_plot))
    plt.close()

    # Print the fit parameters
    print(f'Variance of Δy (Abs Basic LF) fit: m = {m_abs_lf:.4e}')
    print(f'Chi-squared for Δy (Abs Basic LF) fit: {chi_squared_abs_lf:.4f}')

    # -------------------------------
    # 7. Generate Summary Plots for Variance vs. Time
    # -------------------------------

    # Plot all variances on a single plot
    plt.figure(figsize=(10, 6))
    plt.plot(mutation_counts, variance_delta_y_kis, marker='o', label='kis')
    plt.plot(mutation_counts, variance_delta_y_lf, marker='s', label='Basic Local Fields')
    plt.plot(mutation_counts, variance_delta_y_abs_lf / (1 - 2/np.pi), marker='^', label='Absolute Basic Local Fields')
    plt.title('Variance of Δy vs. Number of Mutations')
    plt.xlabel('Number of Mutations')
    plt.ylabel('Variance of Δy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    variance_summary_plot = 'variance_delta_y_vs_mutations_summary.png'
    plt.savefig(os.path.join(main_output_dir, subdirs['variance_plots'], variance_summary_plot))
    plt.close()

    print(f"Variance vs. Number of Mutations summary plot saved as {variance_summary_plot}")

    # -------------------------------
    # 8. Generate Summary Plots for Correlations vs. Rank
    # -------------------------------

    # Function to plot summary correlations
    def plot_summary_correlations(df_all, correlation_type='pearson'):
        """
        Plots summary correlations vs. rank.

        Parameters:
        - df_all (pd.DataFrame): DataFrame containing all correlation data.
        - correlation_type (str): 'pearson' or 'spearman'.
        """
        plt.figure(figsize=(10, 6))

        # Define mapping for column names based on correlation type
        if correlation_type == 'pearson':
            cols = {
                'kis': 'pearson_r_kis',
                'basic_local_fields': 'pearson_r_local_fields',
                'absolute_basic_local_fields': 'pearson_r_abs_local_fields'
            }
            ylabel = 'Pearson r'
        elif correlation_type == 'spearman':
            cols = {
                'kis': 'spearman_r_kis',
                'basic_local_fields': 'spearman_r_local_fields',
                'absolute_basic_local_fields': 'spearman_r_abs_local_fields'
            }
            ylabel = 'Spearman rho'
        else:
            raise ValueError("correlation_type must be either 'pearson' or 'spearman'.")

        # Plot each correlation type
        for label, col in cols.items():
            plt.plot(df_all.index, df_all[col], marker='o', label=label.replace('_', ' ').title())

        # Set title and labels
        plt.title(f'{ylabel} Correlations vs. Rank')
        plt.xlabel('Rank')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)

        # Ensure the x-axis goes from largest to smallest rank
        plt.gca().invert_xaxis()

        # Save the summary plot
        summary_plot_filename = f'{correlation_type}_correlations_vs_rank.png'
        plt.tight_layout()
        plt.savefig(os.path.join(main_output_dir, subdirs['correlation_summary_plots'], summary_plot_filename))
        plt.close()

        print(f"{correlation_type.title()} Correlations vs. Rank plot saved as {summary_plot_filename}")

    # Check if the correlation coefficients CSV exists
    correlation_csv_path = os.path.join(main_output_dir, 'correlation_coefficients.csv')
    if os.path.exists(correlation_csv_path):
        df_all = pd.read_csv(correlation_csv_path, index_col='rank')

        # Plot Pearson correlations vs. rank
        plot_summary_correlations(df_all, correlation_type='pearson')

        # Plot Spearman correlations vs. rank
        plot_summary_correlations(df_all, correlation_type='spearman')
    else:
        print("Correlation coefficients CSV not found. Summary plots not generated.")


if __name__ == "__main__":
    main()
