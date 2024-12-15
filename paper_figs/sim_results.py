import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from misc.uncmn_dfe import gen_final_dfe
from scipy.stats import gaussian_kde
import matplotlib.ticker as ticker
import misc.uncmn_scrambling as scr
from misc import cmn_sk
import scienceplots
from matplotlib.gridspec import GridSpec

# Define a consistent style
plt.style.use('science')
plt.rcParams['font.family'] = 'Helvetica Neue'
file_path = '../misc/run_data/N4000_rho100_beta100_repeats50.pkl'
color = sns.color_palette('CMRmap', 5)

with open(file_path, 'rb') as f:
    data = pickle.load(f)


def create_fig_ge(ax, num_points, repeat, N):
    # same as before
    if num_points > 100:
        raise ValueError("The number of points must be less than 100.")
    data_entry = data[repeat]
    alpha_initial = data_entry['init_alpha']
    h = data_entry['h']
    J = data_entry['J']
    F_off = cmn.calc_F_off(alpha_initial, h, J)
    flip_seq = data_entry['flip_seq']
    flip_numbs = np.linspace(0, len(flip_seq)-1, num_points, dtype=int)
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
        mean_dfe.append(np.mean(dfe) + 1)
        var_dfe.append(np.var(dfe))
        mean_bdfe.append(np.mean(bdfe))
        var_bdfe.append(np.var(bdfe))
        rank.append(cmn.calc_rank(alpha, h, J))

    rank = [r / N for r in rank]
    max_fit = fits[-1]
    fits = [fit * 100 / max_fit for fit in fits]
    sns.regplot(x=fits, y=mean_dfe, ax=ax, color=color[0], scatter=True, label=r'$\mathbb{E} [P(\Delta)] + 1$')
    sns.regplot(x=fits, y=var_dfe, ax=ax, color=color[1], scatter=True, label=r'$\text{Var} [P(\Delta)]$')
    sns.regplot(x=fits, y=mean_bdfe, ax=ax, color=color[2], scatter=True, label=r'$\mathbb{E} [P_+ (\Delta)]$')
    sns.regplot(x=fits, y=var_bdfe, ax=ax, color=color[3], scatter=True, label=r'$\text{Var} [P_+ (\Delta)]$')
    sns.regplot(x=fits, y=rank, ax=ax, color=color[4], scatter=True, label='$r(t) / N$')
    ax.set_xlabel('Fitness (\\% from maximum reached)', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.legend(fontsize=10, title_fontsize=12, loc='lower left', frameon=True)


def create_fig_dfe_fin(ax, N, beta_arr, rho, num_repeats, num_bins):
    # same as before
    for i, beta in enumerate(beta_arr):
        dfe = gen_final_dfe(N, beta, rho, num_repeats)
        hist, bin_edges = np.histogram(dfe, bins=num_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.scatter(bin_centers, hist, label=f'$\\beta ={beta_arr[i]:.1f}$', color=color[i % len(color)],
                   edgecolor=color[i % len(color)], s=5, alpha=0.6)
        ax.plot(bin_centers, hist, color=color[i % len(color)], lw=2, alpha=0.6)

    ax.set_xlabel('$\\Delta$', fontsize=14)
    ax.set_ylabel('$P(\\Delta)$', fontsize=14)
    ax.set_xlim(None, 0)
    ax.legend(fontsize=12, title_fontsize=12, loc='upper left', bbox_to_anchor=(0.075, 0.925), frameon=True, markerscale=3)

    # Make tick lines
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both', nbins=5))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune='both', nbins=4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    current_ticks = ax.get_xticks()
    new_ticks = np.concatenate(([0.0], current_ticks))
    ax.set_xticks(new_ticks)
    ax.set_xticklabels(['0.0'] + [f'{tick:.1f}' for tick in current_ticks])


def create_fig_bdfe_hists(ax, points_lst, num_bins, num_flips):
    # same as before
    num = len(points_lst)
    bdfes = [[] for _ in range(num)]
    del_max = 0.3
    for repeat in data:
        alphas = cmn.curate_alpha_list(repeat['init_alpha'], repeat['flip_seq'], points_lst)
        h = repeat['h']
        J = repeat['J']
        for j in range(num):
            bdfe = cmn.calc_BDFE(alphas[j], h, J)[0]
            bdfes[j].extend(bdfe[bdfe < del_max])

    flip_percent = (points_lst / num_flips) * 100
    for i, bdfe in enumerate(bdfes):
        hist, bin_edges = np.histogram(bdfe, bins=num_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        label = f'{flip_percent[i]:.0f}$\\%$'
        ax.scatter(bin_centers, hist, label=label, color=color[i % len(color)], edgecolor=color[i % len(color)], s=5, alpha=0.6)
        ax.plot(bin_centers, hist, color=color[i % len(color)], lw=2, alpha=0.6)

    ax.set_xlabel('$\\Delta$', fontsize=14)
    ax.set_ylabel('$log \\left( P_+ (\\Delta) \\right)$', fontsize=14)
    ax.set_xlim(0, None)
    ax.set_yscale('log')

    # Adjust legend
    ax.legend(title='$\\%$ of walk \ncompleted', fontsize=12, title_fontsize=12, loc='lower left',
              frameon=True, markerscale=3)

    # Set major and minor x-ticks automatically
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune='both', nbins=3))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Set major and minor y-ticks automatically with pruning
    ax.yaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=2))
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    current_ticks = ax.get_xticks()
    xlim = ax.get_xlim()
    new_ticks = np.concatenate(([xlim[0]], current_ticks))
    ax.set_xticks(new_ticks)
    ax.set_xticklabels([f'{xlim[0]:.1f}'] + [f'{tick:.1f}' for tick in current_ticks])


def create_fig_crossings(ax, flip1, flip2, repeat):
    # same as before
    data_entry = data[repeat]
    alpha_initial = data_entry['init_alpha']
    h = data_entry['h']
    J = data_entry['J']
    flip_seq = data_entry['flip_seq']
    color1 = color[0]
    color2 = color[2]
    scr.gen_crossings(ax, alpha_initial, h, J, flip_seq, flip1, flip2, color1, color2)


def create_fig_e(ax_left, ax_right, flip1, flip2, repeat, color):
    data_entry = data[repeat]
    alpha_initial = data_entry['init_alpha']
    h = data_entry['h']
    J = data_entry['J']
    flip_seq = data_entry['flip_seq']

    # Forward and backward propagation
    bdfe1, prop_bdfe1, dfe_evo = scr.propagate_forward(alpha_initial, h, J, flip_seq, flip1, flip2)
    bdfe2, prop_bdfe2, dfe_anc = scr.propagate_backward(alpha_initial, h, J, flip_seq, flip1, flip2)

    ax_left.clear()
    ax_right.clear()

    # Left subplot:
    sns.kdeplot(dfe_evo, ax=ax_left, color='black', fill=True, alpha=0.6)
    sns.kdeplot(bdfe1, ax=ax_left, color=color[1], fill=True, alpha=0.6)
    sns.kdeplot(prop_bdfe2, ax=ax_left, color=color[2], fill=True, alpha=0.6)

    # Right subplot:
    sns.kdeplot(dfe_anc, ax=ax_right, color='black', fill=True, alpha=0.6)
    sns.kdeplot(bdfe2, ax=ax_right, color=color[2], fill=True, alpha=0.6)
    sns.kdeplot(prop_bdfe1, ax=ax_right, color=color[1], fill=True, alpha=0.6)

    # Vertical dotted lines at x=0
    ax_left.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax_right.axvline(x=0, color='black', linestyle='--', linewidth=1)

    # Only show a tick at 0 for x
    ax_left.set_xticks([0])
    ax_left.set_xticklabels(['0'])
    ax_right.set_xticks([0])
    ax_right.set_xticklabels(['0'])

    # Previously we removed y-ticks from the right axis. Now let's keep them:
    # Remove these lines: ax_right.set_yticks([]), ax_right.set_yticklabels([])
    # The right axis shares y with the left, so it will have the same tick positions.
    # We just remove labels on the right side:
    ax_right.yaxis.set_ticks_position('right')
    ax_right.tick_params(axis='y', labelleft=False, labelright=False)

    # Remove spines to merge visually
    ax_left.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)

    # Remove ticks on the right side of the left axis
    ax_left.yaxis.set_ticks_position('left')
    ax_left.tick_params(labelright=False, right=False)

    # Common Y label on the left subplot
    ax_left.set_ylabel("$P(\\Delta)$", fontsize=14)
    ax_right.set_ylabel("")

    # Styling of ticks
    ax_left.tick_params(axis='both', which='major', length=10, width=1, labelsize=14)
    ax_left.tick_params(axis='both', which='minor', length=5, width=1, labelsize=14)
    ax_right.tick_params(axis='both', which='major', length=10, width=1, labelsize=14)
    ax_right.tick_params(axis='both', which='minor', length=5, width=1, labelsize=14)


if __name__ == "__main__":
    N = 4000
    num_flips = int(N * 0.64)
    high = int(num_flips * 0.8)
    low = int(num_flips * 0.6)
    flip_list = np.linspace(low, high, 4, dtype=int)
    crossings_repeat = 10
    crossings_flip_anc = 800
    crossings_flip_evo = 1200

    big_fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 3, figure=big_fig)

    # Top row: A, B, C
    axA = big_fig.add_subplot(gs[0, 0])
    axB = big_fig.add_subplot(gs[0, 1])
    axC = big_fig.add_subplot(gs[0, 2])

    # Bottom row: D and E
    axD = big_fig.add_subplot(gs[1, 0])
    gs_e = gs[1, 1:].subgridspec(1, 2, wspace=0)
    axE_left = big_fig.add_subplot(gs_e[0, 0])
    axE_right = big_fig.add_subplot(gs_e[0, 1], sharey=axE_left)

    # Tweak tick params for consistency
    for ax in [axA, axB, axC, axD, axE_left, axE_right]:
        ax.tick_params(axis='both', which='major', length=10, width=1, labelsize=14)
        ax.tick_params(axis='both', which='minor', length=5, width=1, labelsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    # create_fig_ge(axA, num_points=50, repeat=10, N=N)
    # create_fig_dfe_fin(axB, N=2000, beta_arr=[0.0001, 0.5, 1.0], rho=1.0, num_repeats=3, num_bins=100)
    # create_fig_bdfe_hists(axC, points_lst=flip_list, num_bins=50, num_flips=num_flips)
    # create_fig_crossings(axD, crossings_flip_anc, crossings_flip_evo, crossings_repeat)
    create_fig_e(axE_left, axE_right, flip1=crossings_flip_anc, flip2=crossings_flip_evo, repeat=crossings_repeat, color=color)

    # Remove the x=0 label from subplots A-D (if they have ticks and a '0' label):
    for ax in [axA, axB, axC, axD]:
        xticks = ax.get_xticks()
        # Replace the label at 0 with an empty string
        new_labels = []
        for t in xticks:
            if np.isclose(t, 0.0):
                new_labels.append('')
            else:
                # Format other ticks normally
                new_labels.append(f'{t:.1f}')
        ax.set_xticklabels(new_labels)

    # Label the panels
    panel_labels = ['A', 'B', 'C', 'D', 'E']
    ax_list = [axA, axB, axC, axD, axE_left]
    for i, ax in enumerate(ax_list):
        ax.text(-0.1, 1.1, panel_labels[i], transform=ax.transAxes,
                fontsize=16, fontweight='heavy', va='top', ha='left')

    big_fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    output_dir = '../Plots/paper_figs'
    os.makedirs(output_dir, exist_ok=True)
    big_fig.savefig(os.path.join(output_dir, "sim_results_modified.png"), dpi=600)