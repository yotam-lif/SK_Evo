import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from code_sim.cmn.uncmn_dfe import gen_final_dfe, propagate_forward, propagate_backward
import matplotlib.ticker as ticker
from code_sim.cmn import cmn, cmn_sk
from matplotlib.gridspec import GridSpec
import statsmodels.api as sm
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib as mpl

# Define a consistent style
# plt.style.use('science')
plt.rcParams['font.family'] = 'sans-serif'
file_path = '../code_sim/data/SK/N4000_rho100_beta100_repeats50.pkl'
color = sns.color_palette('CMRmap', 5)
# Global font settings for labels, ticks, and legends
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 12

def reflect_kde_neg(data, bw='scott', kernel='gau', gridsize=200):
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

with open(file_path, 'rb') as f:
    data = pickle.load(f)

def create_fig_dfe_evol(ax, num_points, repeat):
    # Pull out the relevant data
    data_entry = data[repeat]
    sigma_initial = data_entry['init_alpha']
    h = data_entry['h']
    J = data_entry['J']
    flip_seq = data_entry['flip_seq']

    # Timeslices (in percent) along the adaptive walk
    percents = np.linspace(0, 100, num_points, dtype=int)
    ts = [int(len(flip_seq) * pct / 100) for pct in percents]
    sigma_list = cmn.curate_sigma_list(sigma_initial, flip_seq, ts)

    # Compute DFEs at each time slice
    dfes = [cmn_sk.compute_dfe(sigma, h, J) for sigma in sigma_list]

    area_ref = 1.0  # Will hold total area of the first distribution (from histogram)
    for i, dfe in enumerate(dfes):
        # First, get the “raw” histogram area for the i-th DFE
        counts, bin_edges = np.histogram(dfe, bins=40, density=False)
        dx = bin_edges[1] - bin_edges[0]
        area_i = counts.sum() * dx  # total area in the unnormalized histogram

        # If this is the first distribution, store its area as our reference
        if i == 0:
            area_ref = area_i  # we’ll scale everything else to this

        # Now, do a KDE on the same data. By default, the KDE integrates to 1.
        kde = sm.nonparametric.KDEUnivariate(dfe)
        kde.fit(kernel='gau', bw='scott', fft=False, gridsize=200, cut=3)
        # The integral of kde.density over kde.support is 1.
        # We want to scale it so that the *total area* becomes area_i/area_ref.
        scale_factor = (area_i / area_ref)  # ensures we replicate the same ratio
        scaled_density = kde.density * scale_factor

        # Finally, plot
        ax.plot(kde.support,
                scaled_density + 0.003,
                lw=2,
                color=color[i % len(color)],
                label=f'$t={percents[i]}\\%$'
               )

    ax.set_xlabel(r'Fitness effect $(\Delta)$')
    ax.set_ylabel(r'$P(\Delta, t)$')
    ax.set_ylim(0, 0.35)
    ax.axvline(x=0, color='black', linestyle=':', lw=1.5)
    ax.legend(loc='upper left', frameon=False, markerscale=3)

def create_fig_dfe_fin(ax, N, beta_arr, rho, num_repeats):
    # same as before
    for i, beta in enumerate(beta_arr):
        dfe = gen_final_dfe(N, beta, rho, num_repeats)
        x_kde, y_kde = reflect_kde_neg(dfe, bw='scott', kernel='gau', gridsize=200)
        ax.plot(x_kde, y_kde, label=f'$\\beta={beta_arr[i]:.1f}$', color=color[i % len(color)], lw=2.0)
    ax.set_xlabel(r'Fitness effect $(\Delta)$')
    ax.set_ylabel(r'$P(\Delta, t=100\%)$')
    ax.set_xlim(None, 0)
    ax.legend(loc='upper left', frameon=False, markerscale=3)

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
        label = f'$t={flip_percent[i]:.0f}\\%$'
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

    ax.set_xlabel(r'Fitness effect $(\Delta)$')
    ax.set_ylabel(r'$ln(P_+(\Delta, t))$')
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
    ax.legend(loc='upper right', frameon=False)

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

    x_left = flip_anc_percent
    x_right = flip_evo_percent
    alpha = 0.3

    # Draw arrows for forward propagation: from left (ancestral) to right (proposed)
    for j in range(len(bdfe1)):
        start = (x_left, bdfe1[j])
        end = (x_right, prop_bdfe1[j])
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10,
                                color=color1, lw=0.75, alpha=alpha)
        ax.add_patch(arrow)
    # Scatter plot the forward points
    ax.scatter(np.repeat(x_left, len(bdfe1)), bdfe1, color=color1, edgecolor=color1,
               s=20, facecolors='none', label='Forwards', alpha=alpha)
    ax.scatter(np.repeat(x_right, len(prop_bdfe1)), prop_bdfe1, color=color1, edgecolor=color1,
               s=20, facecolors='none', alpha=alpha)

    # Draw arrows for backward propagation: from right (proposed) to left (evolved)
    for j in range(len(bdfe2)):
        start = (x_right, bdfe2[j])
        end = (x_left, prop_bdfe2[j])
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10,
                                color=color2, lw=0.75, alpha=alpha)
        ax.add_patch(arrow)
    # Scatter plot the backward points
    ax.scatter(np.repeat(x_left, len(prop_bdfe2)), prop_bdfe2, color=color2, edgecolor=color2,
               s=20, facecolors='none', label='Backwards', alpha=alpha)
    ax.scatter(np.repeat(x_right, len(bdfe2)), bdfe2, color=color2, edgecolor=color2,
               s=20, facecolors='none', alpha=alpha)

    # Set x-axis to show the flip percentages
    ax.set_xticks([x_left, x_right])
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel('$\\%$ of walk completed')
    ax.set_ylabel(r'Fitness effect $(\Delta)$')
    ax.legend(frameon=False, loc='upper right')

def create_fig_dfes_overlap(ax_left, ax_right, flip1, flip2, repeat, color):
    # Get simulation data parameters from the global data variable.
    data_entry = data[repeat]
    alpha_initial = data_entry['init_alpha']
    h = data_entry['h']
    J = data_entry['J']
    flip_seq = data_entry['flip_seq']

    # Calculate percentages of walk completed like in create_fig_crossings
    num_flips = len(flip_seq)
    flip_anc_percent = int(flip1 * 100 / num_flips)
    flip_evo_percent = int(flip2 * 100 / num_flips)

    fraction_z = 0.08  # change as needed
    lw_main = 1.0

    # Propagate forward/backward.
    bdfe_anc, prop_bdfe_anc, dfe_evo = propagate_forward(alpha_initial, h, J, flip_seq, flip1, flip2)
    bdfe_evo, prop_bdfe_evo, dfe_anc = propagate_backward(alpha_initial, h, J, flip_seq, flip1, flip2)

    # --- Helper: scale x from reference range [-0.1, 0.1] to our fixed x–limits.
    def scale_x(x_ref, X_min, X_max, ref_min=-0.1, ref_max=0.1):
        return X_min + ((x_ref - ref_min) / (ref_max - ref_min)) * (X_max - X_min)

    # --- Custom segments drawing function.
    def draw_custom_segments(ax, X_min, X_max, y_bottom, z_val, lw_main):
        x_left_line = scale_x(-0.09, X_min, X_max)
        x_right_line = scale_x(0.09, X_min, X_max)
        ax.plot([x_left_line, x_right_line], [z_val, z_val],
                linestyle="--", color="grey", lw=lw_main)
        segs_ref = [
            ((-0.05, y_bottom), ((-0.05 * 0.9), z_val)),
            ((0.05, y_bottom), ((0.05 * 0.9), z_val)),
            ((-0.1, y_bottom), (-0.09, z_val)),
            ((0.1, y_bottom), (0.09, z_val)),
            ((0, y_bottom), (0, z_val))
        ]
        for (pt0, pt1) in segs_ref:
            x0 = scale_x(pt0[0], X_min, X_max)
            x1 = scale_x(pt1[0], X_min, X_max)
            ax.plot([x0, x1], [pt0[1], pt1[1]], linestyle="--", color="grey", lw=lw_main)

    # Define fill colors.
    EVO_FILL = (color[1][0], color[1][1], color[1][2], 0.75)
    ANC_FILL = (0.5, 0.5, 0.5, 0.4)

    # ========================
    # LEFT PANEL
    # ========================
    ax_left.set_ylim(0, 1.2)
    left_X_min, left_X_max = -10, 10
    ax_left.set_xlim(left_X_min, left_X_max)
    ax_left.set_xlabel(r'Fitness effect $(\Delta)$')
    ax_left.tick_params()

    counts, bin_edges = np.histogram(prop_bdfe_anc, bins=20, density=True)
    y_max_unshifted = ax_left.get_ylim()[1]
    z = fraction_z * y_max_unshifted
    ax_left.stairs(
        values=counts + z,
        edges=bin_edges,
        baseline=0,
        fill=True,
        facecolor=EVO_FILL,
        edgecolor="black",
        lw=1.1,
        label=f'Evolved(${flip_evo_percent}\\%$)',
        zorder=0
    )
    y_bottom_left = -0.01
    eps_height = 0.005
    eps_width = 0.1
    rect_left = Rectangle((left_X_min - eps_width, y_bottom_left - eps_height),
                          (left_X_max - left_X_min) + 2 * eps_width,
                          (z - y_bottom_left) + 2 * eps_height,
                          facecolor="white", edgecolor="none", zorder=2)
    ax_left.add_patch(rect_left)

    anc_counts, anc_bin_edges = np.histogram(bdfe_anc, bins=10, density=True)
    anc_bin_edges += 0.5

    dfe_evo_counts, dfe_evo_bin_edges = np.histogram(prop_bdfe_anc, bins=20, density=True)
    ax_left.stairs(
        values=dfe_evo_counts + z - 0.01,
        edges=dfe_evo_bin_edges + 0.2,
        baseline=0,
        fill=False,
        facecolor=None,
        edgecolor=color[2],
        lw=1.1,
        label=f'Full DFE(${flip_evo_percent}\\%$)',
        zorder=1
    )

    ax_left.stairs(
        values=anc_counts,
        edges=anc_bin_edges,
        baseline=0,
        fill=True,
        facecolor=ANC_FILL,
        edgecolor="black",
        lw=1.1,
        label=f'Ancestor(${flip_anc_percent}\\%$)',
        zorder=3
    )
    ax_left.figure.canvas.draw()
    ax_left.legend(frameon=False, loc='upper left', fontsize=10)
    draw_custom_segments(ax_left, left_X_min, left_X_max, y_bottom_left, z, lw_main)


    # ========================
    # RIGHT PANEL
    # ========================
    right_X_min, right_X_max = -10, 10
    ax_right.set_xlim(right_X_min, right_X_max)
    ax_right.set_xlabel(r'Fitness effect $(\Delta)$')
    ax_right.tick_params()

    counts2, bin_edges2 = np.histogram(bdfe_evo, bins=8, density=True)
    bin_edges2 += 0.5
    y_max_unshifted_right = ax_right.get_ylim()[1]
    z_right = fraction_z * y_max_unshifted_right
    ax_right.cla()
    ax_right.set_xlim(right_X_min, right_X_max)
    ax_right.set_xlabel(r'Fitness effect $(\Delta)$')
    ax_right.tick_params()
    ax_right.stairs(
        values=counts2 + z_right,
        edges=bin_edges2,
        baseline=0,
        fill=True,
        facecolor=EVO_FILL,
        edgecolor="black",
        lw=1.1,
        label=f'Evolved(${flip_evo_percent}\\%$)'
    )
    ax_right.figure.canvas.draw()
    y_bottom_right = -0.01
    rect_right = Rectangle((right_X_min - eps_width, y_bottom_right - eps_height),
                           (right_X_max - right_X_min) + 2 * eps_width,
                           (z_right - y_bottom_right) + 2 * eps_height,
                           facecolor="white", edgecolor="none")
    ax_right.add_patch(rect_right)
    draw_custom_segments(ax_right, right_X_min, right_X_max, y_bottom_right, z_right, lw_main)

    anc2_counts, anc2_bin_edges = np.histogram(prop_bdfe_evo, bins=15, density=True)
    ax_right.stairs(
        values=anc2_counts,
        edges=anc2_bin_edges,
        baseline=0,
        fill=True,
        facecolor=ANC_FILL,
        edgecolor="black",
        lw=1.1,
        label=f'Ancestor(${flip_anc_percent}\\%$)'
    )
    dfe_anc2_counts, dfe_anc2_bin_edges = np.histogram(dfe_anc, bins=18, density=True)
    ax_right.stairs(
        values=dfe_anc2_counts,
        edges=dfe_anc2_bin_edges,
        baseline=0,
        fill=False,
        facecolor=None,
        edgecolor=color[2],
        lw=1.1,
        label=f'Full DFE(${flip_anc_percent}\\%$)'
    )

    ax_right.legend(frameon=False, loc='upper left', fontsize=10)
    ax_right.set_ylim(0, None)

    for ax in [ax_left, ax_right]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(width=1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
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

    # Increase figure size and adjust spacing to avoid overlap
    big_fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=big_fig, wspace=0.4, hspace=0.4)

    # Top row: A, B, C
    axA = big_fig.add_subplot(gs[0, 0])
    axB = big_fig.add_subplot(gs[0, 1])
    axC = big_fig.add_subplot(gs[0, 2])

    # Bottom row: D and a subgrid for E and F
    axD = big_fig.add_subplot(gs[1, 0])
    gs_e = gs[1, 1:].subgridspec(1, 2, wspace=0.3)
    axE = big_fig.add_subplot(gs_e[0, 0])
    axF = big_fig.add_subplot(gs_e[0, 1], sharey=axE)

    # Tweak tick parameters and match tick width to spine width
    for ax in [axA, axB, axC, axD, axE, axF]:
        ax.tick_params(axis='both', which='major', length=10, width=1.5, labelsize=14)
        ax.tick_params(axis='both', which='minor', length=5, width=1.6, labelsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    create_fig_dfe_evol(axA, num_points=5, repeat=crossings_repeat)
    create_fig_dfe_fin(axB, N=2000, beta_arr=[0.0, 0.5, 1.0], rho=1.0, num_repeats=3)
    create_fig_bdfe_hists(axC, points_lst=flip_list, num_bins=10, num_flips=num_flips)
    create_fig_crossings(axD, crossings_flip_anc, crossings_flip_evo, crossings_repeat)
    create_fig_dfes_overlap(axE, axF, flip1=crossings_flip_anc, flip2=crossings_flip_evo, repeat=crossings_repeat, color=color)

    # Remove the x=0 label from subplots B, C, D as before
    for ax in [axB, axC, axD]:
        xticks = ax.get_xticks()
        new_labels = [f'{t:.1f}' if not np.isclose(t, 0.0) else '' for t in xticks]
        ax.set_xticklabels(new_labels)

    # Label the panels: now we have 6 panels (A, B, C, D, E, F)
    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    ax_list = [axA, axB, axC, axD, axE, axF]
    for i, ax in enumerate(ax_list):
        ax.text(-0.1, 1.1, panel_labels[i], transform=ax.transAxes, fontweight='heavy', va='top', ha='left')

    big_fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_dir = '../figs_paper'
    os.makedirs(output_dir, exist_ok=True)
    big_fig.savefig(os.path.join(output_dir, "sk_results.svg"), format="svg", bbox_inches='tight')
