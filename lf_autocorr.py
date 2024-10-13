import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr  # For calculating correlation coefficients
import pandas as pd  # For saving correlation data

# Import the Funcs module
import Funcs as Fs


def main():
    # -------------------------------
    # 1. Initialize SK Environment
    # -------------------------------

    # Parameters
    N = 2000  # Number of spins
    beta = 0.25  # Epistasis strength
    rho = 0.05  # Fraction of non-zero coupling elements
    random_state = 42  # Seed for reproducibility

    # Initialize spin configuration
    alpha_initial = Fs.init_alpha(N)

    # Initialize external fields
    h = Fs.init_h(N, beta=beta, random_state=random_state)

    # Initialize coupling matrix with sparsity
    J = Fs.init_J(N, beta=beta, rho=rho, random_state=random_state)

    # Define desired ranks using linear spacing
    num_points = 150  # Adjust this number as needed
    max_rank = 1000
    min_rank = 0

    # Generate linearly spaced ranks descendingly from max_rank to min_rank
    ranks = np.linspace(max_rank, min_rank, num=num_points).astype(int)

    # Remove any duplicates and ensure descending order
    ranks = np.unique(ranks)[::-1]

    print(f"\nInitial Rank: {max_rank}")
    print(f"Ranks to Save (Descending Order): {ranks}\n")

    # Relax the system using sswm_flip (sswm=True)
    # Now returns flips (number of mutations up to each rank)
    final_alpha, saved_alphas, flips = Fs.relax_SK(alpha_initial.copy(), h, J, ranks, sswm=True)

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
        'fitness_effects_correlation': 'fitness_effects_correlation',
        'local_fields_correlation': 'local_fields_correlation',
        'absolute_local_fields_correlation': 'absolute_local_fields_correlation',
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
    correlation_fitness_effects = []
    correlation_local_fields = []
    correlation_absolute_local_fields = []

    mutation_counts = flips  # This serves as our "time" variable

    # -------------------------------
    # 4. Set Initial Values from Max Rank
    # -------------------------------

    # Assuming the first saved_alpha corresponds to max_rank
    if saved_alphas and saved_alphas[0] is not None:
        alpha_initial_correlation = saved_alphas[0]
        kis_initial = Fs.calc_kis(alpha_initial_correlation, h, J)
        fitness_effects_initial = -2 * kis_initial
        basic_local_fields_initial = Fs.calc_basic_lfs(alpha_initial_correlation, h, J)
        abs_basic_local_fields_initial = np.abs(basic_local_fields_initial)

        print("\n--- Correlation Initial Values ---")
        print("Initial Correlation Spin Configuration set from max_rank.")
        print("-----------------------------------\n")
    else:
        raise ValueError("No valid alpha configuration found at max_rank for correlation analyses.")

    # -------------------------------
    # 5. Iterate Over Ranks and Generate Correlation Plots
    # -------------------------------

    for idx, (rank, alpha) in enumerate(zip(ranks, saved_alphas)):
        if alpha is not None:
            # -------------------------------
            # 5.1 Calculate Current Values
            # -------------------------------

            # Calculate current kis values
            kis_current = Fs.calc_kis(alpha, h, J)

            # Calculate current fitness effects
            fitness_effects_current = -2 * kis_current

            # Calculate current basic local fields
            basic_local_fields_current = Fs.calc_basic_lfs(alpha, h, J)

            # Calculate current absolute basic local fields
            abs_basic_local_fields_current = np.abs(basic_local_fields_current)

            # -------------------------------
            # 5.2 kis Correlation
            # -------------------------------

            # Compute Pearson correlation coefficient for kis
            pearson_r_kis, p_value_kis = pearsonr(kis_initial, kis_current)

            # Compute Spearman correlation coefficient for kis
            spearman_r_kis, p_value_spearman_kis = spearmanr(kis_initial, kis_current)

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
            sns.scatterplot(x=kis_initial, y=kis_current, alpha=0.5, edgecolor=None, s=20)

            # Plot reference lines
            max_val_kis = max(np.max(kis_initial), np.max(kis_current))
            min_val_kis = min(np.min(kis_initial), np.min(kis_current))
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
            # 5.3 Fitness Effects Correlation
            # -------------------------------

            # Compute Pearson correlation coefficient for fitness effects
            pearson_r_fitness, p_value_fitness = pearsonr(fitness_effects_initial, fitness_effects_current)

            # Compute Spearman correlation coefficient for fitness effects
            spearman_r_fitness, p_value_spearman_fitness = spearmanr(fitness_effects_initial, fitness_effects_current)

            # Append fitness effects correlation data
            correlation_fitness_effects.append({
                'rank': rank,
                'pearson_r_fitness_effects': pearson_r_fitness,
                'p_value_fitness_effects': p_value_fitness,
                'spearman_r_fitness_effects': spearman_r_fitness,
                'p_value_spearman_fitness_effects': p_value_spearman_fitness
            })

            # Set up the fitness effects correlation plot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=fitness_effects_initial, y=fitness_effects_current, alpha=0.5, edgecolor=None, s=20)

            # Plot reference lines
            max_val_fitness = max(np.max(fitness_effects_initial), np.max(fitness_effects_current))
            min_val_fitness = min(np.min(fitness_effects_initial), np.min(fitness_effects_current))
            plt.plot([min_val_fitness, max_val_fitness], [min_val_fitness, max_val_fitness],
                     color='red', linestyle='--', label='y = x')
            plt.plot([min_val_fitness, max_val_fitness], [-min_val_fitness, -max_val_fitness],
                     color='green', linestyle='--', label='y = -x')

            # Annotate with Pearson and Spearman correlations
            plt.text(0.05, 0.95,
                     f'Pearson r = {pearson_r_fitness:.4f}\nSpearman rho = {spearman_r_fitness:.4f}',
                     transform=plt.gca().transAxes,
                     fontsize=12,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            # Title and labels
            plt.title(f'Fitness Effects Correlation at Rank {rank}; N={N}, β={beta}, ρ={rho}')
            plt.xlabel('Initial Fitness Effects')
            plt.ylabel(f'Fitness Effects at Rank {rank}')
            plt.legend()
            plt.tight_layout()

            # Save the fitness effects correlation plot
            plot_filename_fitness = f'correlation_fitness_effects_rank_{rank}.png'
            plt.savefig(os.path.join(main_output_dir, subdirs['fitness_effects_correlation'], plot_filename_fitness))
            plt.close()

            print(f"Fitness Effects Correlation scatter plot for rank {rank} saved as {plot_filename_fitness}")

            # -------------------------------
            # 5.4 Basic Local Fields Correlation
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
            # 5.5 Absolute Basic Local Fields Correlation
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
    # 6. Save All Correlation Data to CSV
    # -------------------------------

    # After processing all ranks
    if correlation_kis and correlation_fitness_effects and correlation_local_fields and correlation_absolute_local_fields:
        # Merge all correlation data based on ranks
        df_kis = pd.DataFrame(correlation_kis).set_index('rank')
        df_fitness = pd.DataFrame(correlation_fitness_effects).set_index('rank')
        df_lf = pd.DataFrame(correlation_local_fields).set_index('rank')
        df_abs_lf = pd.DataFrame(correlation_absolute_local_fields).set_index('rank')

        # Combine all DataFrames
        df_all = df_kis.join(df_fitness).join(df_lf).join(df_abs_lf)

        # Sort the DataFrame by rank in descending order
        df_all = df_all.sort_index(ascending=False)

        # Define the CSV filename
        correlation_filename = 'correlation_coefficients.csv'

        # Save to CSV in the main output directory
        df_all.to_csv(os.path.join(main_output_dir, correlation_filename))
        print(f"\nCorrelation coefficients saved as {correlation_filename} in '{main_output_dir}' directory.")
    else:
        print("\nNo correlation data to save.")

    # -------------------------------
    # 7. Generate Summary Plots for Correlations vs. Rank
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
                'fitness_effects': 'pearson_r_fitness_effects',
                'basic_local_fields': 'pearson_r_local_fields',
                'absolute_basic_local_fields': 'pearson_r_abs_local_fields'
            }
            ylabel = 'Pearson r'
        elif correlation_type == 'spearman':
            cols = {
                'kis': 'spearman_r_kis',
                'fitness_effects': 'spearman_r_fitness_effects',
                'basic_local_fields': 'spearman_r_local_fields',
                'absolute_basic_local_fields': 'spearman_r_abs_local_fields'
            }
            ylabel = 'Spearman rho'
        else:
            raise ValueError("correlation_type must be either 'pearson' or 'spearman'.")

        # Plot each correlation type
        # also reverse the x-axis to show the highest rank first
        for label, col in cols.items():
            plt.plot(df_all.index, df_all[col], marker='o', label=label.replace('_', ' ').title())

        plt.gca().invert_xaxis()
        # Set title and labels
        plt.title(f'{ylabel} Correlations vs. Rank')
        plt.xlabel('Rank')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)

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

        # Sort the DataFrame by rank in descending order to ensure correct plotting
        df_all = df_all.sort_index(ascending=False)

        # Plot Pearson correlations vs. rank
        plot_summary_correlations(df_all, correlation_type='pearson')

        # Plot Spearman correlations vs. rank
        plot_summary_correlations(df_all, correlation_type='spearman')
    else:
        print("Correlation coefficients CSV not found. Summary plots not generated.")


if __name__ == "__main__":
    main()