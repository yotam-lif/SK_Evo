import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.stats import pearsonr

def main():
    # Configuration
    FITNESS_LOWER_THRESHOLD = -0.5  # Include genes with fitness effect > -0.5
    FITNESS_UPPER_THRESHOLD = 0.1    # Exclude genes with fitness effect > 0.1

    # Load the data
    data = np.load('fitness_corrected_genes.npy')  # Shape: (gene, replicate, time)

    # Determine the number of genes, replicates, and time points
    num_genes, num_replicates, num_timepoints = data.shape

    if num_timepoints != 2:
        raise ValueError(f"Expected 2 time points, but got {num_timepoints}")

    # Define the directory for saving plots
    script_dir = Path(__file__).parent  # Directory where the script is located
    plots_dir = script_dir / "plots"

    # Create the plots directory if it doesn't exist
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Define subdirectory for scatter plots
    scatter_plots_dir = plots_dir / "scatter_plots"
    scatter_plots_dir.mkdir(parents=True, exist_ok=True)

    # Define colors and labels for the two time points
    colors = ['grey', 'skyblue']  # Updated colors: grey and sky blue
    time_labels = [f"Time {t}" for t in range(num_timepoints)]
    time_generations = [0, 50000]  # Generation 0 and t=50k generations

    # Iterate over each replicate
    for rep in range(num_replicates):
        # Extract data for the current replicate across all genes and both time points
        replicate_data = data[:, rep, :]  # Shape: (gene, time)

        # Identify genes where both time points have FITNESS_LOWER_THRESHOLD < effect <= FITNESS_UPPER_THRESHOLD
        valid_genes = np.all(
            (replicate_data > FITNESS_LOWER_THRESHOLD) & (replicate_data <= FITNESS_UPPER_THRESHOLD),
            axis=1
        )

        # Exclude invalid genes
        valid_data = replicate_data[valid_genes, :]  # Shape: (valid_gene, time)

        # If no valid genes remain, skip plotting for this replicate
        if valid_data.size == 0:
            print(f"No genes with {FITNESS_LOWER_THRESHOLD} < fitness effect <= {FITNESS_UPPER_THRESHOLD} for Replicate {rep + 1}. Skipping plot.")
            continue

        # Prepare the data for each time point
        time0_data = valid_data[:, 0]
        time1_data = valid_data[:, 1]

        # -------------------------------
        # Histogram Plotting
        # -------------------------------

        # Create a new figure for histograms
        plt.figure(figsize=(10, 6))

        # Define the number of bins (adjustable as needed)
        bins = 50

        # Plot histograms for each time point with updated colors
        plt.hist(
            time0_data,
            bins=bins,
            alpha=0.4,
            color=colors[0],
            label=time_labels[0],
            edgecolor='black',
            density=True
        )
        plt.hist(
            time1_data,
            bins=bins,
            alpha=0.4,
            color=colors[1],
            label=time_labels[1],
            edgecolor='black',
            density=True
        )

        # Add a vertical line at 0
        plt.axvline(0, color='black', linestyle='solid', linewidth=1.5, label='Zero Effect')

        # Calculate and plot mean fitness effects
        mean_time0 = np.mean(time0_data)
        mean_time1 = np.mean(time1_data)

        plt.axvline(mean_time0, color=colors[0], linestyle='dashed', linewidth=1, label=f'Mean {time_labels[0]}: {mean_time0:.2f}')
        plt.axvline(mean_time1, color=colors[1], linestyle='dashed', linewidth=1, label=f'Mean {time_labels[1]}: {mean_time1:.2f}')

        # Annotate mean lines
        plt.text(mean_time0, plt.ylim()[1]*0.9, f'Mean {time_labels[0]}: {mean_time0:.2f}',
                 color=colors[0], ha='center', fontsize=9)
        plt.text(mean_time1, plt.ylim()[1]*0.8, f'Mean {time_labels[1]}: {mean_time1:.2f}',
                 color=colors[1], ha='center', fontsize=9)

        # Add title and labels
        plt.title(f'Fitness Effects - Replicate {rep + 1}')
        plt.xlabel('Fitness Effect')
        plt.ylabel('Density')

        # Add legend, ensuring 'Zero Effect' is included
        plt.legend()

        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save the histogram plot to the plots directory
        hist_plot_filename = plots_dir / f'replicate_{rep + 1}_histogram.png'
        plt.savefig(hist_plot_filename)
        plt.close()  # Close the figure to free memory

        print(f"Saved histogram for Replicate {rep + 1} to {hist_plot_filename}")

        # -------------------------------
        # Scatter Plotting
        # -------------------------------

        # Create a new figure for scatter plot
        plt.figure(figsize=(8, 8))

        # Scatter plot of fitness effect at time 0 vs time 1
        plt.scatter(time0_data, time1_data, alpha=0.6, edgecolor='k', linewidth=0.5)

        # Add identity line (y = x)
        min_val = min(time0_data.min(), time1_data.min())
        max_val = max(time0_data.max(), time1_data.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

        # Calculate Pearson correlation coefficient
        if len(time0_data) > 1:
            corr_coef, p_value = pearsonr(time0_data, time1_data)
            corr_text = f'Pearson r = {corr_coef:.2f}\nP-value = {p_value:.2e}'
        else:
            corr_text = 'Not enough data for correlation'

        # Add correlation coefficient text to the plot
        plt.text(0.05, 0.95, corr_text, transform=plt.gca().transAxes,
                 verticalalignment='top', fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

        # Add title and labels
        plt.title(f'Fitness Effect Correlation - Replicate {rep + 1}')
        plt.xlabel('Fitness Effect at Time 0 (Generation 0)')
        plt.ylabel('Fitness Effect at Time 1 (t=50k Generations)')

        # Add legend
        plt.legend()

        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save the scatter plot to the scatter_plots subdirectory
        scatter_plot_filename = scatter_plots_dir / f'replicate_{rep + 1}_scatter.png'
        plt.savefig(scatter_plot_filename)
        plt.close()  # Close the figure to free memory

        print(f"Saved scatter plot for Replicate {rep + 1} to {scatter_plot_filename}")

    print("All plots have been generated and saved.")

if __name__ == "__main__":
    main()