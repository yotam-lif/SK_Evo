import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from misc.uncmn_dfe import gen_final_dfe
from scipy.stats import gaussian_kde
import matplotlib.ticker as ticker
import misc.uncmn_scrambling as scr
from misc import cmn
import scienceplots

# Define a consistent style
plt.style.use('science')
plt.rcParams['font.family'] = 'Helvetica Neue'
file_path = '../misc/run_data/N4000_rho100_beta100_repeats50.pkl'
color = sns.color_palette('CMRmap', 4)

with open(file_path, 'rb') as f:
    data = pickle.load(f)


def create_fig_ge(ax, num_points, repeat, N):
    """
    Parameters:
        ax (plt.Axes): The matplotlib axis to plot on.
        num_points (int): The number of points to consider for the plot.
        repeat (int): The repeat index to consider.
        N (int): The system size to consider.

    Returns:
        None
    """
    print("\n create_fig_ge called")
    if num_points > 100:
        raise ValueError("The number of points must be less than 100.")
    data_entry = data[repeat]
    alpha_initial = data_entry['init_alpha']
    h = data_entry['h']
    J = data_entry['J']
    F_off = cmn.calc_F_off(alpha_initial, h, J)
    flip_seq = data_entry['flip_seq']
    flip_numbs = np.linspace(0, len(flip_seq), num_points, dtype=int)
    alphas = cmn.curate_alpha_list(alpha_initial, flip_seq, flip_numbs)
    mean_dfe = []
    var_dfe = []
    mean_bdfe = []
    var_bdfe = []
    rank = []
    fits = []
    for alpha in alphas:
        dfe = cmn.calc_DFE(alpha, h, J)
        bdfe = cmn.calc_BDFE(alpha, h, J)[0]
        fits.append(cmn.compute_fit_slow(alpha, h, J, F_off))
        mean_dfe.append(np.mean(dfe))
        var_dfe.append(np.var(dfe))
        mean_bdfe.append(np.mean(bdfe))
        var_bdfe.append(np.var(bdfe))
        rank.append(cmn.calc_rank(alpha, h, J))

    rank = [r / N for r in rank]
    max_fit = fits[-1]
    fits = [fit * 100 / max_fit for fit in fits]
    sns.regplot(x=fits, y=mean_bdfe, ax=ax, color=color[0], scatter=True, label='Mean BDFE')
    sns.regplot(x=fits, y=var_dfe, ax=ax, color=color[1], scatter=True, label='Var DFE')
    sns.regplot(x=fits, y=var_bdfe, ax=ax, color=color[2], scatter=True, label='Var BDFE')
    sns.regplot(x=fits, y=rank, ax=ax, color=color[3], scatter=True, label='$r(t) / N$')
    ax.set_xlabel('F(t) (\\% from maximum reached)', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.legend(fontsize=12, title_fontsize=12, loc='upper right', bbox_to_anchor=(0.925, 0.925), frameon=True)

def create_fig_dfe_fin(ax, N, beta_arr, rho, num_repeats, num_bins):
    """
    Creates DFE histograms on the given Axes.

    Parameters:
        ax (plt.Axes): The matplotlib axis to plot on.
        N (int): The system size to consider.
        beta_arr (list of floats): The beta parameter list for the DFE.
        rho (float): The rho parameter for the DFE.
        num_repeats (int): The number of repeats to consider.
        num_bins (int): The number of bins to use in the histogram.

    Returns:
        None

    """
    print("\n create_fig_dfe_fin called")
    for i, beta in enumerate(beta_arr):
        dfe = gen_final_dfe(N, beta, rho, num_repeats)
        hist, bin_edges = np.histogram(dfe, bins=num_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # kde = gaussian_kde(dfe)
        ax.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], alpha=0.6, label=f'$\\beta ={beta_arr[i]:.1f}$', color=color[i % len(color)], edgecolor=color[i % len(color)])
        # ax.plot(bin_centers, kde(bin_centers), color=color[-(i % len(color))], lw=2, alpha=0.4)

    ax.set_xlabel('$\\Delta$', fontsize=14)
    ax.set_ylabel('$P(\\Delta)$', fontsize=14)
    ax.set_xlim(None, 0)
    ax.legend(fontsize=12, title_fontsize=12, loc='upper left', bbox_to_anchor=(0.075, 0.925), frameon=True)

    # Make tick lines
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both', nbins=5))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune='both', nbins=4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    current_ticks = ax.get_xticks()
    new_ticks = np.concatenate(([0.0], current_ticks))  # Add the limits to the current ticks
    ax.set_xticks(new_ticks)  # Set the new tick positions
    # Label the ticks at the limits
    ax.set_xticklabels(['0.0'] + [f'{tick:.1f}' for tick in current_ticks])


def create_fig_bdfe_hists(ax, points_lst, num_bins, num_flips):
    """
    Creates BDFE histograms on the given Axes.

    Parameters:
        ax (plt.Axes): The matplotlib axis to plot on.
        points_lst (np.ndarray): The list of points to consider for the BDFE.
        num_bins (int): The number of bins to use in the histogram.
        num_flips (int): The total number of flips in the system.

    Returns:
        None
    """
    print("\n create_fig_bdfe_hists called")
    num = len(points_lst)
    bdfes = [[] for _ in range(num)]
    for repeat in data:
        alphas = cmn.curate_alpha_list(repeat['init_alpha'], repeat['flip_seq'], points_lst)
        h = repeat['h']
        J = repeat['J']
        for j in range(num):
            bdfes[j].extend(cmn.calc_BDFE(alphas[j], h, J)[0])

    flip_percent = (points_lst / num_flips) * 100

    # Store the labels for the legend
    for i, bdfe in enumerate(bdfes):
        # hist, bin_edges = np.histogram(bdfe, bins=num_bins, density=True)
        # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # label = f'{flip_percent[i]:.0f}$\\%$'  # Format the label for percentage
        # ax.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], alpha=0.6, label=label, color=color[i % len(color)])
        sns.kdeplot(bdfe, ax=ax, label=f'{flip_percent[i]:.0f}$\\%$', color=color[i % len(color)], fill=True, alpha=0.5)

    ax.set_xlabel('$\\Delta$', fontsize=14)
    ax.set_ylabel('$log \\left( P_+ (\\Delta) \\right)$', fontsize=14)
    ax.set_xlim(0, None)
    ax.set_yscale('log')

    # Adjust legend
    ax.legend(title='$\\%$ of flips', fontsize=12, title_fontsize=12, loc='upper right', bbox_to_anchor=(0.925, 0.925),
              frameon=True)

    # Set major and minor x-ticks automatically
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune='both', nbins=3))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Set major and minor y-ticks automatically with pruning
    ax.yaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=2))
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    # Get the current ticks and limits
    current_ticks = ax.get_xticks()
    xlim = ax.get_xlim()
    # Add ticks at the limits without removing the existing ones
    new_ticks = np.concatenate(([xlim[0]], current_ticks))  # Add the limits to the current ticks
    ax.set_xticks(new_ticks)  # Set the new tick positions
    # Label the ticks at the limits
    ax.set_xticklabels([f'{xlim[0]:.1f}'] + [f'{tick:.1f}' for tick in current_ticks])


def create_fig_crossings(ax, flip1, flip2, repeat):
    """
    Generate and plot crossing data.

    Parameters:
        ax (plt.Axes): The matplotlib axis to plot on.
        flip1 (int): The first flip to consider.
        flip2 (int): The second flip to consider.
        repeat (int): The repeat index.

    Returns:
        None
    """
    print("\n create_fig_crossings called")
    data_entry = data[repeat]
    alpha_initial = data_entry['init_alpha']
    h = data_entry['h']
    J = data_entry['J']
    flip_seq = data_entry['flip_seq']
    color1 = color[0]
    color2 = color[3]
    scr.gen_crossings(ax, alpha_initial, h, J, flip_seq, flip1, flip2, color1, color2)


def create_fig_evo_anc(ax1, ax2, flip1, flip2, repeat):
    """
    Generate and plot evolutionary ancestor data.

    Parameters:
        ax (plt.Axes): The matplotlib axis to plot on.
        flip1 (int): The first flip to consider.
        flip2 (int): The second flip to consider.
        repeat (int): The repeat index.

    Returns:
        None
    """
    print("\n create_fig_evo_anc called")
    data_entry = data[repeat]
    alpha_initial = data_entry['init_alpha']
    h = data_entry['h']
    J = data_entry['J']
    flip_seq = data_entry['flip_seq']
    num_flips = len(flip_seq)

    # Propagate forward and backward for the given flips
    bdfe1, prop_bdfe1, dfe_evo = scr.propagate_forward(alpha_initial, h, J, flip_seq, flip1, flip2)
    bdfe2, prop_bdfe2, dfe_anc = scr.propagate_backward(alpha_initial, h, J, flip_seq, flip1, flip2)

    flip1_percent = int((flip1 / num_flips) * 100)
    flip2_percent = int((flip2 / num_flips) * 100)

    # Plot KDEs for bdfe1 and prop_bdfe1 on ax1
    sns.kdeplot(bdfe1, ax=ax1, label=f'BDFE Anc.', color=color[0], fill=True, alpha=0.6)
    sns.kdeplot(prop_bdfe1, ax=ax1, label=f'Propagated Anc. BDFE', color=color[1], fill=True, alpha=0.6)
    sns.kdeplot(dfe_evo, ax=ax1, label=f'DFE Evo.', color=color[2], fill=True, alpha=0.6)
    ax1.set_xlabel('$\\Delta$', fontsize=14)
    ax1.set_ylabel('$P(\\Delta)$', fontsize=14)
    ax1.legend(fontsize=12, frameon=True)

    # Plot KDEs for bdfe2 and prop_bdfe2 on ax2
    sns.kdeplot(bdfe2, ax=ax2, label=f'BDFE Evo.', color=color[1], fill=True, alpha=0.6)
    sns.kdeplot(prop_bdfe2, ax=ax2, label=f'Propagated Evo. BDFE', color=color[0], fill=True, alpha=0.6)
    sns.kdeplot(dfe_anc, ax=ax2, label=f'DFE Anc.', color=color[2], fill=True, alpha=0.6)
    ax2.set_xlabel('$\\Delta$', fontsize=14)
    ax2.set_ylabel('$P(\\Delta)$', fontsize=14)
    ax2.legend(fontsize=12, frameon=True)

    # Adjust layout
    plt.tight_layout()


if __name__ == "__main__":
    # Create a large figure with subplots
    big_fig, axs = plt.subplots(2, 3, figsize=(12, 10))  # Adjust size for clarity

    # Create each subplot
    N = 4000
    num_flips = int(N * 0.64)
    high = int(num_flips * 0.8)
    low = int(num_flips * 0.6)
    flip_list = np.linspace(low, high, 4, dtype=int)
    crossings_repeat = 10
    crossings_flip_anc = 800
    crossings_flip_evo = 1200

    for ax in axs.flatten():
        ax.tick_params(axis='both', which='major', length=10, width=1, labelsize=14)
        ax.tick_params(axis='both', which='minor', length=5, width=1, labelsize=14)
        # Make the subplot frames thicker
        for spine in ax.spines.values():
            spine.set_linewidth(2)  # Change the thickness of the border

    create_fig_ge(axs[0, 0], num_points=40, repeat=10, N=N)
    create_fig_dfe_fin(axs[0, 1], N=2000, beta_arr=[0.0001, 0.5, 1.0], rho=1.0, num_repeats=3, num_bins=60)
    create_fig_bdfe_hists(axs[0, 2], points_lst=flip_list, num_bins=50, num_flips=num_flips)
    create_fig_crossings(axs[1, 0], crossings_flip_anc, crossings_flip_evo, crossings_repeat)
    create_fig_evo_anc(axs[1, 1], axs[1, 2], crossings_flip_anc, crossings_flip_evo, crossings_repeat)

    # Panel labels (A-E)
    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    for i, ax in enumerate(axs.flatten()):
        ax.text(-0.1, 1.1, panel_labels[i], transform=ax.transAxes,
                fontsize=16, fontweight='heavy', va='top', ha='left')

    # Adjust layout for spacing and a central title
    big_fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    output_dir = '../Plots/paper_figs'
    os.makedirs(output_dir, exist_ok=True)
    big_fig.savefig(os.path.join(output_dir, "sim_results.png"), dpi=600)
