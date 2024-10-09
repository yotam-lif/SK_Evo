import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr  # For calculating correlation coefficients
import pandas as pd  # For saving correlation data

# Import the Funcs module
import Funcs as Fs


def main():
    # -------------------------------
    # 1. Initialize SK Environment
    # -------------------------------

    # Parameters
    N = 2000  # Number of spins
    beta = 0.75  # Epistasis strength
    rho = 0.1  # Fraction of non-zero coupling elements
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
    final_alpha, saved_alphas = Fs.relax_SK(alpha_initial.copy(), h, J, ranks, sswm=True)

    # Calculate initial kis values
    ki_initial = Fs.calc_kis(alpha_initial, h, J)

    # Calculate initial basic local fields
    basic_local_fields_initial = h + J.dot(alpha_initial)

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

    # Define subdirectories for each correlation type
    subdirs = {
        'kis_correlation': 'kis_correlation',
        'local_fields_correlation': 'local_fields_correlation',
        'absolute_local_fields_correlation': 'absolute_local_fields_correlation'
    }

    # Create subdirectories
    for key, subdir in subdirs.items():
        path = os.path.join(main_output_dir, subdir)
        os.makedirs(path, exist_ok=True)

    # -------------------------------
    # 3. Initialize Lists to Store Correlation Data
    # -------------------------------

    correlation_kis = []
    correlation_local_fields = []
    correlation_absolute_local_fields = []

    # -------------------------------
    # 4. Iterate Over Ranks and Generate Plots
    # -------------------------------

    for rank, alpha in zip(ranks, saved_alphas):
        if alpha is not None:
            # -------------------------------
            # 4.1 Calculate Current Values
            # -------------------------------

            # Calculate current kis values
            ki_current = Fs.calc_kis(alpha, h, J)

            # Calculate current basic local fields
            basic_local_fields_current = h + J.dot(alpha)

            # Calculate absolute basic local fields
            abs_basic_local_fields_initial = np.abs(basic_local_fields_initial)
            abs_basic_local_fields_current = np.abs(basic_local_fields_current)

            # -------------------------------
            # 4.2 kis Correlation
            # -------------------------------

            # Compute Pearson correlation coefficient for kis
            correlation_kis_val, p_value_kis = pearsonr(ki_initial, ki_current)

            # Append kis correlation data
            correlation_kis.append({
                'rank': rank,
                'pearson_r_kis': correlation_kis_val,
                'p_value_kis': p_value_kis
            })

            # Set up the kis correlation plot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=ki_initial, y=ki_current, alpha=0.5, edgecolor=None)

            # Plot a reference line (y = x) for better visualization
            max_val_kis = max(np.max(ki_initial), np.max(ki_current))
            min_val_kis = min(np.min(ki_initial), np.min(ki_current))
            plt.plot([min_val_kis, max_val_kis], [min_val_kis, max_val_kis], color='red', linestyle='--', label='y = x')

            # Annotate with Pearson r
            plt.text(0.05, 0.95, f'Pearson r = {correlation_kis_val:.4f}',
                     transform=plt.gca().transAxes,
                     fontsize=12,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            # Title and labels for kis correlation
            plt.title(f'kis Correlation at Rank {rank}; N={N}, β={beta}, ρ={rho}')
            plt.xlabel('Initial kis')
            plt.ylabel(f'kis at Rank {rank}')
            plt.legend()

            # Save the kis correlation plot in its subdirectory
            plot_filename_kis = f'correlation_kis_rank_{rank}.png'
            plt.tight_layout()
            plt.savefig(os.path.join(main_output_dir, subdirs['kis_correlation'], plot_filename_kis))
            plt.close()

            print(f"kis Correlation scatter plot for rank {rank} saved as {plot_filename_kis}")

            # -------------------------------
            # 4.3 Basic Local Fields Correlation
            # -------------------------------

            # Compute Pearson correlation coefficient for basic local fields
            correlation_local_fields_val, p_value_local_fields = pearsonr(basic_local_fields_initial,
                                                                          basic_local_fields_current)

            # Append basic local fields correlation data
            correlation_local_fields.append({
                'rank': rank,
                'pearson_r_local_fields': correlation_local_fields_val,
                'p_value_local_fields': p_value_local_fields
            })

            # Set up the basic local fields correlation plot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=basic_local_fields_initial, y=basic_local_fields_current, alpha=0.5, edgecolor=None)

            # Plot a reference line (y = x) for better visualization
            max_val_lf = max(np.max(basic_local_fields_initial), np.max(basic_local_fields_current))
            min_val_lf = min(np.min(basic_local_fields_initial), np.min(basic_local_fields_current))
            plt.plot([min_val_lf, max_val_lf], [min_val_lf, max_val_lf], color='red', linestyle='--', label='y = x')

            # Annotate with Pearson r
            plt.text(0.05, 0.95, f'Pearson r = {correlation_local_fields_val:.4f}',
                     transform=plt.gca().transAxes,
                     fontsize=12,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            # Title and labels for basic local fields correlation
            plt.title(f'Basic Local Fields Correlation at Rank {rank}; N={N}, β={beta}, ρ={rho}')
            plt.xlabel('Initial Basic Local Fields')
            plt.ylabel(f'Basic Local Fields at Rank {rank}')
            plt.legend()

            # Save the basic local fields correlation plot in its subdirectory
            plot_filename_lf = f'correlation_local_fields_rank_{rank}.png'
            plt.tight_layout()
            plt.savefig(os.path.join(main_output_dir, subdirs['local_fields_correlation'], plot_filename_lf))
            plt.close()

            print(f"Basic Local Fields Correlation scatter plot for rank {rank} saved as {plot_filename_lf}")

            # -------------------------------
            # 4.4 Absolute Basic Local Fields Correlation
            # -------------------------------

            # Compute Pearson correlation coefficient for absolute basic local fields
            correlation_abs_lf_val, p_value_abs_lf = pearsonr(abs_basic_local_fields_initial,
                                                              abs_basic_local_fields_current)

            # Append absolute basic local fields correlation data
            correlation_absolute_local_fields.append({
                'rank': rank,
                'pearson_r_abs_local_fields': correlation_abs_lf_val,
                'p_value_abs_local_fields': p_value_abs_lf
            })

            # Set up the absolute basic local fields correlation plot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=abs_basic_local_fields_initial, y=abs_basic_local_fields_current, alpha=0.5,
                            edgecolor=None)

            # Plot a reference line (y = x) for better visualization
            max_val_abs_lf = max(np.max(abs_basic_local_fields_initial), np.max(abs_basic_local_fields_current))
            min_val_abs_lf = min(np.min(abs_basic_local_fields_initial), np.min(abs_basic_local_fields_current))
            plt.plot([min_val_abs_lf, max_val_abs_lf], [min_val_abs_lf, max_val_abs_lf], color='red', linestyle='--',
                     label='y = x')

            # Annotate with Pearson r
            plt.text(0.05, 0.95, f'Pearson r = {correlation_abs_lf_val:.4f}',
                     transform=plt.gca().transAxes,
                     fontsize=12,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            # Title and labels for absolute basic local fields correlation
            plt.title(f'Absolute Basic Local Fields Correlation at Rank {rank}; N={N}, β={beta}, ρ={rho}')
            plt.xlabel('Initial |Basic Local Fields|')
            plt.ylabel(f'|Basic Local Fields| at Rank {rank}')
            plt.legend()

            # Save the absolute basic local fields correlation plot in its subdirectory
            plot_filename_abs_lf = f'correlation_absolute_local_fields_rank_{rank}.png'
            plt.tight_layout()
            plt.savefig(
                os.path.join(main_output_dir, subdirs['absolute_local_fields_correlation'], plot_filename_abs_lf))
            plt.close()

            print(
                f"Absolute Basic Local Fields Correlation scatter plot for rank {rank} saved as {plot_filename_abs_lf}")

    # Save all correlation data to CSV after processing all ranks
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


if __name__ == "__main__":
    main()
