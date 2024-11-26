import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dfe.dfe_cmn import gen_bdfes
from scipy.optimize import curve_fit
import scienceplots
# Import the Funcs module
from misc import cmn


plt.style.use('science')


def linear_fit(x, a, b):
    """
    Linear function for fitting: y = a * x + b
    """
    return a * x + b


def create_fig_dfe_fin(ax, N, beta, rho, num_bins, num_points, num_repeats):
    """
    Creates the DFE Final plot on the given Axes.
    """
    ax.set_title("DFE Final", fontsize=16)
    ax.plot([0, 1], [0, 1], label='Placeholder Line')  # Placeholder plot
    ax.set_xlabel('X-axis Label', fontsize=14)
    ax.set_ylabel('Y-axis Label', fontsize=14)
    ax.legend()


def create_fig_bdfe_hists(ax, N, beta, rho, num_points, num_repeats, num_bins):
    """
    Creates BDFE histograms on the given Axes.
    Each histogram is plotted for the log-transformed BDFE values.

    Parameters:
    - ax: Matplotlib Axes object to plot on.
    - N: Parameter N.
    - beta: Parameter beta.
    - rho: Parameter rho.
    - num_points: Number of data points.
    - num_repeats: Number of repeats.
    - num_bins: Number of histogram bins.
    """
    # Generate BDFE histograms
    bdfes = gen_bdfes(N, beta, rho, num_points, num_repeats)

    # Define the range of BDFEs to plot (e.g., 60% to 80%)
    low = int(num_points * 0.5)
    up = int(num_points * 0.7)
    real_num_points = up - low

    # Define flip percentages for labeling
    flip_percent = np.linspace(0, 100, num_points)

    # Generate a color palette with as many colors as selected BDFEs
    pal = sns.cubehelix_palette(real_num_points, rot=-.2)

    # Loop through each selected BDFE, apply log transform, and plot histogram
    for idx, i in enumerate(range(low, up)):
        bdfe_i = bdfes[i]

        # Create histogram with density normalization
        hist, bin_edges = np.histogram(bdfe_i, bins=num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot histogram as bars
        ax.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0],
               color=pal[idx], alpha=0.5, label=f'${flip_percent[i]:.0f} % $')

    # Customize the plot
    ax.set_title(f'BDFE Evolution', fontsize=16)
    ax.set_xlabel('$\\Delta$', fontsize=14)
    ax.set_ylabel('$P(\\Delta)$', fontsize=14)
    ax.set_yscale('log')  # Log scale for y-axis
    ax.set_xlim(left=0)  # Set the x-axis lower limit to 0

    # Add a legend with Seaborn's style
    ax.legend(title='% of flips from total', fontsize=10, title_fontsize=12, loc='upper right')

    # Adjust grid spacing for better readability
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    # ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))  # Sparse grid every 0.5
    # ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    # sns.despine(trim=True)


def create_fig_evo_anc(ax, N, beta, rho, num_bins, num_points, num_repeats):
    """
    Creates the Evolution Ancestor plot on the given Axes.
    """
    ax.set_title("Evolution Ancestor", fontsize=16)
    ax.scatter(np.random.rand(10), np.random.rand(10), label='Placeholder Scatter')  # Placeholder scatter
    ax.set_xlabel('X-axis Label', fontsize=14)
    ax.set_ylabel('Y-axis Label', fontsize=14)
    ax.legend()


if __name__ == "__main__":
    # Create a big figure with subplots
    big_fig, axs = plt.subplots(2, 2, figsize=(24, 20))  # Adjusted figsize for clarity

    # Generate each subplot
    # Uncomment the lines below to include other plots as needed
    # create_fig_dfe_fin(axs[0, 0], N, beta, rho, num_bins, num_points, num_repeats)
    create_fig_bdfe_hists(axs[0, 1], N=1500, beta=1.0, rho=1.0, num_points=20, num_repeats=2, num_bins=50)
    # create_fig_crossings(axs[1, 0], N, beta, rho, num_bins, num_points, num_repeats)
    # create_fig_evo_anc(axs[1, 1], N, beta, rho, num_bins, num_points, num_repeats)

    # Set a main title
    big_fig.suptitle("Combined Figure", fontsize=24)

    # Adjust layout to accommodate the main title
    big_fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for the suptitle

    # Save the big figure
    output_dir = '../Plots/paper_figs'
    os.makedirs(output_dir, exist_ok=True)
    big_fig.savefig(os.path.join(output_dir, "sim_results.png"), dpi=600)
    plt.show()
