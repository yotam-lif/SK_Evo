import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from misc.uncmn_dfe import gen_bdfes, gen_final_dfes
import matplotlib.ticker as ticker
import scienceplots


# Define a consistent style
plt.style.use('science')
plt.rcParams['font.family'] = 'Helvetica Neue'
sns.color_palette(palette='YlGnBu')

def create_fig_dfe_fin(ax, N_arr, beta, rho, num_repeats, num_bins):
    print("\n create_fig_dfe_fin called")
    dfes = gen_final_dfes(N_arr, beta, rho, num_repeats)
    pal = sns.cubehelix_palette(len(N_arr), rot=-.2, reverse=True)
    for idx, dfe in enumerate(dfes):
        hist, bin_edges = np.histogram(dfe, bins=num_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], alpha=0.5, label=f'N={N_arr[idx]}')

    ax.set_xlabel('$\\Delta$', fontsize=14)
    ax.set_ylabel('$P(\\Delta)$', fontsize=14)
    ax.set_xlim(None, 0)
    ax.legend(fontsize=12, title_fontsize=12, loc='upper left', bbox_to_anchor=(0.075, 0.925))

    # Make the subplot frames thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Change the thickness of the border

    # Make tick lines
    ax.tick_params(axis='both', which='major', length=20, width=1, labelsize=14)
    ax.tick_params(axis='both', which='minor', length=10, width=1, labelsize=14)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both', nbins=5))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune='both', nbins=4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    current_ticks = ax.get_xticks()
    new_ticks = np.concatenate(([0.0], current_ticks))  # Add the limits to the current ticks
    ax.set_xticks(new_ticks)  # Set the new tick positions
    # Label the ticks at the limits
    ax.set_xticklabels(['0.0'] + [f'{tick:.1f}' for tick in current_ticks])


def create_fig_bdfe_hists(ax, N, beta, rho, num_points, num_repeats, num_bins):
    """
    Creates BDFE histograms on the given Axes.
    """
    print("\n create_fig_bdfe_hists called")
    bdfes = gen_bdfes(N, beta, rho, num_points, num_repeats)
    low, up = int(num_points * 0.5), int(num_points * 0.7)
    real_num_points = up - low
    flip_percent = np.linspace(0, 100, num_points)
    pal = sns.cubehelix_palette(real_num_points, rot=-.2)

    # Store the labels for the legend
    for idx, i in enumerate(range(low, up)):
        bdfe_i = bdfes[i]
        hist, bin_edges = np.histogram(bdfe_i, bins=num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        label = f'{flip_percent[i]:.0f}$\\%$'  # Format the label for percentage

        ax.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], alpha=0.5, label=label)

    ax.set_xlabel('$\\Delta$', fontsize=14)
    ax.set_ylabel('$log \\left( P_+ (\\Delta) \\right)$', fontsize=14)
    ax.set_xlim(0, None)
    ax.set_yscale('log')

    # Adjust legend
    ax.legend(title='$\\%$ of flips', fontsize=12, title_fontsize=12, loc='upper right', bbox_to_anchor=(0.925, 0.925))

    # Make tick lines
    ax.tick_params(axis='both', which='major', length=20, width=1, labelsize=14)
    ax.tick_params(axis='both', which='minor', length=10, width=1, labelsize=14)

    # Set major and minor x-ticks automatically
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune='both', nbins=3))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Set major and minor y-ticks automatically with pruning
    ax.yaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=2))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    # Format the x-axis values (to one decimal point)
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

    # Make the subplot frames thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Change the thickness of the border

    # Get the current ticks and limits
    current_ticks = ax.get_xticks()
    xlim = ax.get_xlim()
    # Add ticks at the limits without removing the existing ones
    new_ticks = np.concatenate(([xlim[0]], current_ticks))  # Add the limits to the current ticks
    ax.set_xticks(new_ticks)  # Set the new tick positions
    # Label the ticks at the limits
    ax.set_xticklabels([f'{xlim[0]:.1f}'] + [f'{tick:.1f}' for tick in current_ticks])


def create_fig_evo_anc(ax):
    """
    Placeholder for Evolution Ancestor plot.
    """
    ax.scatter(np.random.rand(10), np.random.rand(10), label='Placeholder Scatter')
    ax.set_xlabel('X-axis Label', fontsize=12)
    ax.set_ylabel('Y-axis Label', fontsize=12)
    ax.legend()


if __name__ == "__main__":
    # Create a large figure with subplots
    big_fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Adjust size for clarity

    # Create each subplot
    create_fig_dfe_fin(axs[0, 0], N_arr=[1000, 2000], beta=1.0, rho=1.0, num_repeats=2, num_bins=75)
    create_fig_bdfe_hists(axs[0, 1], N=1000, beta=1.0, rho=1.0, num_points=20, num_repeats=2, num_bins=50)
    create_fig_evo_anc(axs[1, 0])

    # Panel labels (A-D)
    panel_labels = ['A', 'B', 'C', 'D']
    for i, ax in enumerate(axs.flatten()):
        ax.text(-0.1, 1.1, panel_labels[i], transform=ax.transAxes,
                fontsize=16, fontweight='heavy', va='top', ha='left')

    # Adjust layout for spacing and a central title
    big_fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    output_dir = '../Plots/paper_figs'
    os.makedirs(output_dir, exist_ok=True)
    big_fig.savefig(os.path.join(output_dir, "sim_results.png"), dpi=600)
