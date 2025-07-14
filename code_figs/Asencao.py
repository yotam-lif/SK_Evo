import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from scipy.stats import ks_2samp

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 12
color = sns.color_palette('CMRmap', 5)
EVO_FILL = (color[1][0], color[1][1], color[1][2], 0.75)
ANC_FILL = (0.5, 0.5, 0.5, 0.4)
DFE_FILL = color[2]
xlim = 0.06
shift_frac = 0.025

datapath = "../data/asencao_dfe_arrays"


def thresholded_histogram(data, threshold, final_bins):
    # Step 1: Use many initial bins to capture fine structure
    init_bins = 10 * final_bins
    counts, bin_edges = np.histogram(data, bins=init_bins)

    # Step 2: Mask bins below threshold
    valid_indices = counts >= threshold
    valid_bin_edges = bin_edges[:-1][valid_indices]
    valid_data = []
    for i, keep in enumerate(valid_indices):
        if keep:
            # Get data in that bin
            bin_mask = (data >= bin_edges[i]) & (data < bin_edges[i+1])
            valid_data.append(data[bin_mask])
    if not valid_data:
        raise ValueError("No bins passed the threshold.")

    # Concatenate all valid data
    cleaned_data = np.concatenate(valid_data)

    # Step 3: Create final histogram with desired number of bins
    final_counts, final_edges = np.histogram(cleaned_data, bins=final_bins, density=True)

    return final_counts, final_edges, cleaned_data


def create_overlapping_dfes(ax_left, ax_right, dfe_anc, dfe_evo):
    # Vertical shift for the "evolved" histograms
    z_frac = 0.1
    lw_main = 1.0

    def draw_custom_segments(ax, xlim, ylim):
        z = ylim * z_frac
        ax.plot([-xlim * 0.9, xlim * 0.9], [z * 1.1, z * 1.1],
                linestyle="--", color="grey", lw=lw_main)
        segs = [
            ((-xlim, -0.75), (-xlim * 0.9, z * 1.1)),
            ((xlim, -0.75), (xlim * 0.9, z * 1.1)),
            ((-xlim / 2, -0.75), (-xlim / 2 * 0.9, z * 1.1)),
            ((xlim / 2, -0.75), (xlim / 2 * 0.9, z * 1.1)),
            ((0, -0.75), (0, z * 1.1))
        ]
        for (x0, y0), (x1, y1) in segs:
            ax.plot([x0, x1], [y0, y1], linestyle="--", color="grey", lw=lw_main)

    bdfe_anc = dfe_anc[dfe_anc > 0]
    bdfe_evo = dfe_evo[dfe_evo > 0]

    bdfe_anc_inds = np.where(dfe_anc > 0)
    bdfe_evo_inds = np.where(dfe_evo > 0)

    prop_bdfe_anc = dfe_evo[bdfe_anc_inds]
    prop_bdfe_evo = dfe_anc[bdfe_evo_inds]

    ks_evo_vs_prop = ks_2samp(dfe_evo, prop_bdfe_anc)
    ks_anc_vs_prop = ks_2samp(dfe_anc, prop_bdfe_evo)

    # Left Panel - Forward propagate
    counts, bin_edges, _ = thresholded_histogram(data=prop_bdfe_anc, threshold=3, final_bins=20)
    anc_counts, anc_bin_edges, _ = thresholded_histogram(data=bdfe_anc, threshold=3, final_bins=20)
    dfe_counts, dfe_bin_edges, _ = thresholded_histogram(data=dfe_evo, threshold=8, final_bins=30)
    max1 = np.max(counts)
    max2 = np.max(anc_counts)
    max3 = np.max(dfe_counts)
    ymax = max(max1, max2, max3)
    ylim = ymax * (1 + z_frac)
    z = ylim * z_frac
    counts_shifted = counts + z
    dfe_counts_shifted = dfe_counts + z
    ax_left.set_xlim(-xlim, xlim)
    ax_left.tick_params(labelsize=14)
    ax_left.set_ylim(0, ylim + 10)
    bin_edges = bin_edges - xlim * shift_frac
    dfe_bin_edges = dfe_bin_edges - xlim * shift_frac
    anc_bin_edges = anc_bin_edges + xlim * shift_frac

    ax_left.stairs(
        values=counts_shifted,
        edges=bin_edges,
        baseline=0,
        fill=True,
        facecolor=EVO_FILL,
        edgecolor="black",
        lw=1.1,
        label="Evolved"
    )

    ax_left.stairs(
        values=dfe_counts_shifted,
        edges=dfe_bin_edges,
        baseline=0,
        fill=False,
        edgecolor=DFE_FILL,
        lw=1.1,
        label="DFE Evo"
    )

    rect = Rectangle((-xlim, 0), 2*xlim, z, facecolor="white", edgecolor="none")
    ax_left.add_patch(rect)
    draw_custom_segments(ax_left, xlim, ylim)

    ax_left.stairs(
        values=anc_counts,
        edges=anc_bin_edges,
        baseline=0,
        fill=True,
        facecolor=ANC_FILL,
        edgecolor="black",
        lw=1.1,
        label="Ancestor"
    )
    ax_left.legend(frameon=False)
    ax_left.set_xlabel(r'Fitness effect $(\Delta)$')
    ax_left.text(
        0.05, 0.95,
        fr'$p_{{KS}} = {ks_evo_vs_prop.pvalue:.2g}$',
        transform=ax_left.transAxes,
        va="top",
        fontsize=12
    )

    # Right Panel
    counts2, bin_edges2, _ = thresholded_histogram(data=bdfe_evo, threshold=2, final_bins=20)
    anc2_counts, anc2_bin_edges, _ = thresholded_histogram(data=prop_bdfe_evo, threshold=2, final_bins=30)
    dfe2_counts, dfe2_bin_edges, _ = thresholded_histogram(data=dfe_anc, threshold=5, final_bins=30)
    max1 = np.max(counts2)
    max2 = np.max(anc2_counts)
    max3 = np.max(dfe2_counts)
    ymax = max(max1, max2, max3)
    ylim = ymax * (1 + z_frac)
    z = ylim * z_frac
    counts2_shifted = counts2 + z
    dfe2_counts_shifted = dfe2_counts + z
    ax_right.set_xlim(-xlim, xlim)
    ax_right.tick_params(labelsize=14)
    bin_edges2 = bin_edges2 + xlim * shift_frac
    dfe2_bin_edges = dfe2_bin_edges - xlim * shift_frac
    anc2_bin_edges = anc2_bin_edges - xlim * shift_frac
    max1 = np.max(counts2_shifted)
    max2 = np.max(anc2_counts)
    max3 = np.max(dfe2_counts_shifted)
    ymax = max(max1, max2, max3)
    ylim = ymax * (1 + z_frac)
    ax_right.set_ylim(0, ylim + 10)

    ax_right.stairs(
        values=counts2_shifted,
        edges=bin_edges2,
        baseline=0,
        fill=True,
        facecolor=EVO_FILL,
        edgecolor="black",
        lw=1.1,
        label="Evolved"
    )
    rect2 = Rectangle((-xlim, 0), 2*xlim, z, facecolor="white", edgecolor="none")
    ax_right.add_patch(rect2)
    draw_custom_segments(ax_right, xlim, ylim)

    ax_right.stairs(
        values=anc2_counts,
        edges=anc2_bin_edges,
        baseline=0,
        fill=True,
        facecolor=ANC_FILL,
        edgecolor="black",
        lw=1.1,
        label="Ancestor"
    )

    ax_right.stairs(
        values=dfe2_counts,
        edges=dfe2_bin_edges,
        baseline=0,
        edgecolor=DFE_FILL,
        lw=1.1,
        label="Anc DFE"
    )

    ax_right.legend(frameon=False)
    ax_right.set_xlabel(r'Fitness effect $(\Delta)$')
    ax_right.text(
        0.05, 0.95,
        fr'$p_{{KS}} = {ks_anc_vs_prop.pvalue:.2g}$',
        transform=ax_right.transAxes,
        va="top",
        fontsize=12
    )

    # Adjust spines and tick positions for a cleaner look
    for ax in [ax_left, ax_right]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

# Find valid experiment directories
experiment_dirs = sorted([
    d for d in os.listdir(datapath)
    if os.path.isdir(os.path.join(datapath, d))
])

for i, exp_dir in enumerate(experiment_dirs):
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)
    exp_path = os.path.join(datapath, exp_dir)
    try:
        R = np.load(os.path.join(exp_path, "R.npy"))
        S = np.load(os.path.join(exp_path, "S.npy"))
        L = np.load(os.path.join(exp_path, "L.npy"))
    except Exception as e:
        print(f"Skipping {exp_dir}: {e}")
        continue

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax_list = [ax1, ax2, ax3, ax4]
    for ax in ax_list:
        ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_visible(True)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-1, 1))  # use scientific notation if outside [1e-2, 1e2]
        ax.tick_params(width=1.5, length=6, which="major")
        ax.tick_params(width=1.5, length=3, which="minor")
        for sp in ax.spines.values():
            sp.set_linewidth(1.5)
        ax.spines["bottom"].set_position(("outward", 10))
        ax.spines["left"].set_position(("outward", 10))
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # remove the indices that are nan in either R or L
    valid_indices = np.isfinite(R) & np.isfinite(L)
    R_valid = R[valid_indices]
    L_valid = L[valid_indices]
    create_overlapping_dfes(ax1, ax2, R_valid, L_valid)
    # remove the indices that are nan in either R or S
    valid_indices = np.isfinite(R) & np.isfinite(S)
    R_valid = R[valid_indices]
    S_valid = S[valid_indices]
    create_overlapping_dfes(ax3, ax4, R_valid, S_valid)
    plt.tight_layout()
    plt.show()