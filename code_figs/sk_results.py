import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from code_sim.cmn.uncmn_dfe import gen_final_dfe, propagate_forward, propagate_backward
import matplotlib.ticker as ticker
from code_sim.cmn import cmn, cmn_sk
from matplotlib.gridspec import GridSpec
from scipy.special import airy
import statsmodels.api as sm
import scienceplots
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Rectangle


def reflect_kde_neg(data, bw='scott', kernel='gau', gridsize=200):
    """
    Perform a boundary-corrected KDE for nonnegative data by reflecting
    about x=0 and fitting a standard (unbounded) KDE to the combined sample.

    Parameters
    ----------
    data : array-like
        1D array of nonnegative data points (Delta >= 0).
    bw : str or float
        Bandwidth specification passed to statsmodels KDEUnivariate.
        Common choices: 'scott', 'silverman', or a numeric value.
    kernel : str
        Kernel name passed to statsmodels (e.g. 'gau' for Gaussian).
    gridsize : int
        Number of grid points for the estimated density.

    Returns
    -------
    x : ndarray
        The support (grid) restricted to x >= 0.
    y : ndarray
        The estimated density on x >= 0.
    """
    # Keep only nonpositive data
    data_pos = np.asarray(data)
    data_pos = data_pos[data_pos <= 0]

    # Reflect about zero
    data_reflected = -data_pos
    data_combined = np.concatenate([data_pos, data_reflected])

    # Fit KDE on the combined sample (which is now symmetric about 0)
    kde = sm.nonparametric.KDEUnivariate(data_combined)
    kde.fit(kernel=kernel, bw=bw, fft=False, gridsize=gridsize, cut=0, clip=(-np.inf, np.inf))
    # (cut=0 ensures we don't pad the domain too far beyond data range)

    # Restrict the support to x <= 0
    mask = (kde.support <= 0)
    x = kde.support[mask]
    # Multiply the density by 2 on x <= 0 because we doubled the sample
    y = 2.0 * kde.density[mask]
    return x, y

# Define a consistent style
# plt.style.use('science')
plt.rcParams['font.family'] = 'sans-serif'
file_path = '../code_sim/data/SK/N4000_rho100_beta100_repeats50.pkl'
color = sns.color_palette('CMRmap', 5)

with open(file_path, 'rb') as f:
    data = pickle.load(f)

def sol_airy(x, D, P_0):
    ai_0 = airy(0)[0]
    c = P_0 / ai_0
    Ai = airy(x / np.sqrt(c * D))[0]
    return (P_0 / ai_0) * Ai

def sol_exp(x, D, P_0):
    return P_0 * np.exp(-x / (P_0 * D))

def create_fig_ge(ax, num_points, repeat, N):
    # same as before
    if num_points > 100:
        raise ValueError("The number of points must be less than 100.")
    data_entry = data[repeat]
    sigma_initial = data_entry['init_alpha']
    h = data_entry['h']
    J = data_entry['J']
    F_off = cmn_sk.compute_fit_off(sigma_initial, h, J)
    flip_seq = data_entry['flip_seq']
    flip_numbs = np.linspace(0, len(flip_seq)-1, num_points, dtype=int)
    alphas = cmn.curate_sigma_list(sigma_initial, flip_seq, flip_numbs)
    mean_dfe = []
    var_dfe = []
    mean_bdfe = []
    var_bdfe = []
    rank = []
    fits = []
    for alpha in alphas:
        dfe = cmn_sk.compute_dfe(alpha, h, J)
        bdfe = cmn_sk.compute_bdfe(alpha, h, J)[0]
        fits.append(cmn_sk.compute_fit_slow(alpha, h, J, F_off))
        mean_dfe.append(np.mean(dfe))
        var_dfe.append(np.var(dfe))
        mean_bdfe.append(np.mean(bdfe))
        var_bdfe.append(np.var(bdfe))
        rank.append(cmn_sk.compute_rank(alpha, h, J))

    rank = [r / N for r in rank]
    max_fit = fits[-1]
    fits = [(fit / max_fit) * 100 for fit in fits]
    sns.regplot(x=fits, y=mean_dfe, ax=ax, color=color[0], scatter=True, label=r'$\mathbb{E} [P(\Delta)]$')
    sns.regplot(x=fits, y=var_dfe, ax=ax, color=color[1], scatter=True, label=r'$\text{Var} [P(\Delta)]$')
    sns.regplot(x=fits, y=mean_bdfe, ax=ax, color=color[2], scatter=True, label=r'$\mathbb{E} [P_+ (\Delta)]$')
    sns.regplot(x=fits, y=var_bdfe, ax=ax, color=color[3], scatter=True, label=r'$\text{Var} [P_+ (\Delta)]$')
    sns.regplot(x=fits, y=rank, ax=ax, color=color[4], scatter=True, label='$r(t) / N$')
    ax.set_xlabel('Fitness (\\% from maximum reached)', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.legend(fontsize=10, title_fontsize=12, loc='lower left', frameon=True)
    # Set x-axis to display integer values
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))


def create_fig_dfe_fin(ax, N, beta_arr, rho, num_repeats, num_bins):
    # same as before
    for i, beta in enumerate(beta_arr):
        dfe = gen_final_dfe(N, beta, rho, num_repeats)
        # hist, bin_edges = np.histogram(dfe, bins=num_bins, density=True)
        # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # ax.scatter(bin_centers, hist, label=f'$\\beta ={beta_arr[i]:.1f}$', color=color[i % len(color)],
        #            edgecolor=color[i % len(color)], s=5, alpha=0.6)
        # ax.plot(bin_centers, hist, color=color[i % len(color)], lw=2, alpha=0.6)
        # sns.histplot(dfe, ax=ax, kde=False, bins=num_bins, label=f'$\\beta={beta_arr[i]:.1f}$', stat='density', element="step", edgecolor=color[i % len(color)], alpha=0.0)
        x_kde, y_kde = reflect_kde_neg(dfe, bw='scott', kernel='gau', gridsize=200)
        ax.plot(x_kde, y_kde, label=f'$\\beta={beta_arr[i]:.1f}$', color=color[i % len(color)], lw=1.5)
    ax.set_xlabel('$\\Delta$', fontsize=14)
    ax.set_ylabel('$P(\\Delta, \\infty)$', fontsize=14)
    ax.set_xlim(None, 0)
    ax.legend(fontsize=12, title_fontsize=12, loc='upper left', frameon=True, markerscale=3)

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
    """
    Plot the BD-FE histograms (log-transformed) for each walk percentage,
    and overlay a dotted line connecting the first bin center to a point
    defined by the third bin (as an example).
    """
    num = len(points_lst)
    bdfes = [[] for _ in range(num)]
    # Collect the BD-FE data for each set of points
    for repeat in data:
        alphas = cmn.curate_sigma_list(repeat['init_alpha'], repeat['flip_seq'], points_lst)
        h = repeat['h']
        J = repeat['J']
        for j in range(num):
            bdfe = cmn_sk.compute_bdfe(alphas[j], h, J)[0]
            bdfes[j].extend(bdfe)

    # Compute percentage labels for each BD-FE set
    flip_percent = (points_lst / num_flips) * 100
    x_min_all, x_max_all = np.inf, -np.inf  # to set a common x-range later

    for i, bdfe in enumerate(bdfes):
        label = f'{flip_percent[i]:.0f}$\\%$'
        # Use a local variable for the bin count rather than updating num_bins directly.
        bins_i = num_bins + (num - i - 1) * 3
        hist, bin_edges = np.histogram(bdfe, bins=bins_i, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Update overall x-range from this histogram
        x_min_all = min(x_min_all, bin_centers[0])
        x_max_all = max(x_max_all, bin_centers[-1])
        # Compute the natural logarithm of the histogram values (shifted to avoid log(0))
        hist_log = np.log(hist + 10**-2)
        # Plot the log-histogram as a step plot.
        ax.step(bin_centers, hist_log, where='mid',
                label=label, color=color[i % len(color)])

        # Instead of a fit, draw a dotted line connecting the first bin center to the third bin center.
        if len(bin_centers) >= 3:
            x0 = bin_centers[0]
            y0 = hist_log[0]
            x1 = bin_centers[2]
            y1 = hist_log[2]
            # Calculate a line between these two points.
            m = (y1 - y0) / (x1 - x0)
            b = y0 - m * x0
            line_vals = m * bin_centers + b
            ax.plot(bin_centers, line_vals, linestyle='--', color=color[i % len(color)])
        else:
            # If there aren't at least 3 bins, just draw a horizontal line at the first value.
            ax.axhline(y=hist_log[0], linestyle='--', color=color[i % len(color)])

    ax.set_xlabel('$\\Delta$', fontsize=14)
    ax.set_ylabel('$\\ln(P_+(\\Delta, t))$', fontsize=14)
    # Set the x-axis limits based on the collected range.
    ax.set_xlim(x_min_all, x_max_all)

    # Use the original (linear) ticker scheme with controlled minor ticks.
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune='both', nbins=3))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune='both', nbins=4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    current_xticks = ax.get_xticks()
    xlim = ax.get_xlim()
    new_xticks = np.concatenate(([xlim[0]], current_xticks))
    ax.set_xticks(new_xticks)
    ax.set_xticklabels([f'{xlim[0]:.1f}'] + [f'{tick:.1f}' for tick in current_xticks])

    # Adjust legend
    ax.legend(title='$\\%$ of walk \ncompleted', fontsize=12, title_fontsize=12, loc='upper right', frameon=True)


def create_fig_crossings(ax, flip1, flip2, repeat):
    data_entry = data[repeat]
    alpha_initial = data_entry['init_alpha']
    h = data_entry['h']
    J = data_entry['J']
    flip_seq = data_entry['flip_seq']
    # Colors for forwards and backwards arrows
    color1 = color[0]  # forwards
    color2 = color[2]  # backwards

    # Compute forward and backward DFEs using flip1 and flip2
    bdfe1, prop_bdfe1, _ = propagate_forward(alpha_initial, h, J, flip_seq, flip1, flip2)
    bdfe2, prop_bdfe2, _ = propagate_backward(alpha_initial, h, J, flip_seq, flip1, flip2)
    num_flips = len(flip_seq)
    flip_anc_percent = int(flip1 * 100 / num_flips)
    flip_evo_percent = int(flip2 * 100 / num_flips)

    # Set fixed x positions for left and right panels (similar to segben.py)
    x_left = 1
    x_right = 2

    # Draw arrows for forward propagation: from left (ancestral) to right (proposed)
    for j in range(len(bdfe1)):
        start = (x_left, bdfe1[j])
        end = (x_right, prop_bdfe1[j])
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10,
                                color=color1, lw=0.75, alpha=0.3)
        ax.add_patch(arrow)
    # Scatter plot the forward points
    ax.scatter(np.repeat(x_left, len(bdfe1)), bdfe1, color=color1, edgecolor=color1,
               s=20, facecolors='none', label='Forwards', alpha=0.3)
    ax.scatter(np.repeat(x_right, len(prop_bdfe1)), prop_bdfe1, color=color1, edgecolor=color1,
               s=20, facecolors='none', alpha=0.3)

    # Draw arrows for backward propagation: from right (proposed) to left (evolved)
    for j in range(len(bdfe2)):
        start = (x_right, bdfe2[j])
        end = (x_left, prop_bdfe2[j])
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10,
                                color=color2, lw=0.75, alpha=0.3)
        ax.add_patch(arrow)
    # Scatter plot the backward points
    ax.scatter(np.repeat(x_left, len(prop_bdfe2)), prop_bdfe2, color=color2, edgecolor=color2,
               s=20, facecolors='none', label='Backwards', alpha=0.3)
    ax.scatter(np.repeat(x_right, len(bdfe2)), bdfe2, color=color2, edgecolor=color2,
               s=20, facecolors='none', alpha=0.3)

    # Set x-axis to show the flip percentages
    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels([f'{flip_anc_percent}\\%', f'{flip_evo_percent}\\%'])

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel('\\% of walk completed', fontsize=14)
    ax.set_ylabel('$\\Delta$', fontsize=14)
    ax.legend(fontsize=12, frameon=True, loc='upper right')


def create_fig_dfes_overlap(ax_left, ax_right, flip1, flip2, repeat, color):
    data_entry = data[repeat]
    alpha_initial = data_entry['init_alpha']
    h = data_entry['h']
    J = data_entry['J']
    flip_seq = data_entry['flip_seq']
    # Vertical shift for the "evolved" histograms
    z = 30
    lw_main = 1.0
    label_fontsize = 16
    tick_fontsize = 14

    # Forward and backward propagation using the current data
    bdfe_anc, prop_bdfe_anc, _ = propagate_forward(alpha_initial, h, J, flip_seq, flip1, flip2)
    bdfe_evo, prop_bdfe_evo, _ = propagate_backward(alpha_initial, h, J, flip_seq, flip1, flip2)

    def draw_custom_segments(ax):
        ax.plot([-0.09, 0.09], [z * 1.1, z * 1.1],
                linestyle="--", color="grey", lw=lw_main)
        segs = [
            ((-0.05, -0.75), (-0.05 * 0.9, z * 1.1)),
            ((0.05, -0.75), (0.05 * 0.9, z * 1.1)),
            ((-0.1, -0.75), (-0.09, z * 1.1)),
            ((0.1, -0.75), (0.09, z * 1.1)),
            ((0, -0.75), (0, z * 1.1))
        ]
        for (x0, y0), (x1, y1) in segs:
            ax.plot([x0, x1], [y0, y1], linestyle="--", color="grey", lw=lw_main)

    PURPLE_FILL = (0.5, 0.0, 0.5, 0.4)  # "purple" but 40% opacity
    GREY_FILL = (0.5, 0.5, 0.5, 0.4)  # "grey"   but 40% opacity

    # --------------
    # Left Panel
    # --------------
    counts, bin_edges = np.histogram(prop_bdfe_anc, bins=30)
    bin_edges = bin_edges * 0.9
    counts_shifted = counts + z

    # ax_left.set_xlim(-0.1, 0.1)
    # ax_left.set_ylim(0, 500)
    ax_left.set_xlabel('Fitness $(\\Delta)$', fontsize=label_fontsize)
    ax_left.tick_params(labelsize=tick_fontsize)
    draw_custom_segments(ax_left)

    # Evolved histogram (purple fill w/ partial alpha, black edge fully opaque)
    ax_left.stairs(
        values=counts_shifted,
        edges=bin_edges,
        baseline=0,
        fill=True,
        facecolor=PURPLE_FILL,  # partially transparent purple
        edgecolor="black",  # fully opaque black
        lw=1.1,
        label="Evolved",
    )

    # Mask bottom portion
    x_min_left = bin_edges[0]
    x_max_left = bin_edges[-1]
    width_left = x_max_left - x_min_left
    eps_height = 0.5
    eps_width = 0.1
    rect = Rectangle((x_min_left - eps_width, -eps_height), width_left + 2 * eps_width, z + eps_height, facecolor="white", edgecolor="none")
    ax_left.add_patch(rect)
    draw_custom_segments(ax_left)

    # Ancestor histogram
    anc_counts, anc_bin_edges = np.histogram(bdfe_anc, bins=15)
    ax_left.stairs(
        values=anc_counts,
        edges=anc_bin_edges,
        baseline=0,
        fill=True,
        facecolor=GREY_FILL,
        edgecolor="black",
        lw=1.1,
        label="Ancestor"
    )

    ax_left.legend(frameon=False, loc='upper left')

    # --------------
    # Right Panel
    # --------------
    counts2, bin_edges2 = np.histogram(bdfe_evo, bins=10)
    bin_edges2 = bin_edges2 * 0.9
    counts2_shifted = counts2 + z

    # ax_right.set_xlim(-0.1, 0.1)
    # ax_right.set_ylim(0, 500)
    ax_right.set_xlabel("Fitness $(\\Delta)$", fontsize=label_fontsize)
    ax_right.tick_params(labelsize=tick_fontsize)

    draw_custom_segments(ax_right)

    # Evolved histogram (purple fill, black edge)
    ax_right.stairs(
        values=counts2_shifted,
        edges=bin_edges2,
        baseline=0,
        fill=True,
        facecolor=GREY_FILL,
        edgecolor="black",
        lw=1.1,
        label="Evolved"
    )

    # Mask bottom portion
    x_min_right = bin_edges2[0]
    x_max_right = bin_edges2[-1]
    width_right = x_max_right - x_min_right
    eps_height = 0.5
    eps_width = 0.1
    rect2 = Rectangle((x_min_right - eps_width, -eps_height), width_right + 2 * eps_width, z + eps_height, facecolor="white", edgecolor="none")
    ax_right.add_patch(rect2)
    draw_custom_segments(ax_right)

    anc2_counts, anc2_bin_edges = np.histogram(prop_bdfe_evo, bins=30)
    ax_right.stairs(
        values=anc2_counts,
        edges=anc2_bin_edges,
        baseline=0,
        fill=True,
        facecolor=PURPLE_FILL,
        edgecolor="black",
        lw=1.1,
        label="Ancestor"
    )

    ax_right.legend(frameon=False, loc='upper left')

    # Detach the axes from the plot region for each subplot
    for ax in [ax_left, ax_right]:
        # Hide the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Offset the bottom spine (x-axis) and left spine (y-axis) outward by 10 points
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))

        # Ensure ticks only appear on the bottom and left spines
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')


if __name__ == "__main__":
    N = 4000
    num_flips = int(N * 0.64)
    high = int(num_flips * 0.9)
    low = int(num_flips * 0.75)
    flip_list = np.linspace(low, high, 4, dtype=int)
    crossings_repeat = 10
    crossings_flip_anc = 400
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
    # create_fig_dfe_fin(axB, N=2000, beta_arr=[0.0, 0.5, 1.0], rho=1.0, num_repeats=3, num_bins=50)
    # create_fig_bdfe_hists(axC, points_lst=flip_list, num_bins=10, num_flips=num_flips)
    # create_fig_crossings(axD, crossings_flip_anc, crossings_flip_evo, crossings_repeat)
    create_fig_dfes_overlap(axE_left, axE_right, flip1=crossings_flip_anc, flip2=crossings_flip_evo, repeat=crossings_repeat, color=color)

    # Remove the x=0 label from subplots:
    for ax in [axB, axC, axD]:
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
    output_dir = '../figs_paper'
    os.makedirs(output_dir, exist_ok=True)
    big_fig.savefig(os.path.join(output_dir, "sk_results.svg"), format="svg", bbox_inches='tight')