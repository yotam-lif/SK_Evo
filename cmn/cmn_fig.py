import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.stats import ks_2samp, cramervonmises_2samp

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


def create_overlapping_dfes(ax_left, ax_right, dfe_anc, dfe_evo, xlim, ANC_FILL, EVO_FILL, DFE_FILL, test=False):
    # Vertical shift for the "evolved" histograms
    z_frac = 0.1
    lw_main = 1.0
    shift_frac = 0.025
    valid_indices = np.isfinite(dfe_anc) & np.isfinite(dfe_evo)
    dfe_anc = dfe_anc[valid_indices]
    dfe_evo = dfe_evo[valid_indices]

    def draw_custom_segments(ax, _xlim, _ylim):
        z = _ylim * z_frac * 1.1
        ax.plot([-_xlim * 0.9, _xlim * 0.9], [z, z],
                linestyle="--", color="grey", lw=lw_main)
        segs = [
            ((-_xlim, -0.75), (-_xlim * 0.9, z)),
            ((_xlim, -0.75), (_xlim * 0.9, z)),
            ((-_xlim / 2, -0.75), (-_xlim / 2 * 0.9, z)),
            ((_xlim / 2, -0.75), (_xlim / 2 * 0.9, z)),
            ((0, -0.75), (0, z))
        ]
        for (x0, y0), (x1, y1) in segs:
            ax.plot([x0, x1], [y0, y1], linestyle="--", color="grey", lw=lw_main)

    bdfe_anc = dfe_anc[dfe_anc > 0]
    bdfe_evo = dfe_evo[dfe_evo > 0]

    bdfe_anc_inds = np.where(dfe_anc > 0)
    bdfe_evo_inds = np.where(dfe_evo > 0)

    prop_bdfe_anc = dfe_evo[bdfe_anc_inds]
    prop_bdfe_evo = dfe_anc[bdfe_evo_inds]

    evo_vs_prop_test = ks_2samp(dfe_evo, prop_bdfe_anc)
    anc_vs_prop_test = ks_2samp(dfe_anc, prop_bdfe_evo)

    # Left Panel - Forward propagate
    counts, bin_edges, _ = thresholded_histogram(data=prop_bdfe_anc, threshold=3, final_bins=25)
    anc_counts, anc_bin_edges, _ = thresholded_histogram(data=bdfe_anc, threshold=3, final_bins=20)
    dfe_counts, dfe_bin_edges, _ = thresholded_histogram(data=dfe_evo, threshold=6, final_bins=30)
    bin_edges = bin_edges - xlim * shift_frac
    dfe_bin_edges = dfe_bin_edges - xlim * shift_frac
    anc_bin_edges = anc_bin_edges + xlim * shift_frac
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

    ax_left.stairs(
        values=counts_shifted,
        edges=bin_edges,
        baseline=0,
        fill=True,
        facecolor=EVO_FILL,
        edgecolor="black",
        lw=1.1,
        label="Evo."
    )

    ax_left.stairs(
        values=dfe_counts_shifted,
        edges=dfe_bin_edges,
        baseline=0,
        fill=False,
        edgecolor=DFE_FILL,
        lw=1.1,
        label="DFE Evo."
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
        label="Anc."
    )
    ax_left.legend(frameon=False)
    ax_left.set_xlabel(r'Fitness effect $(\Delta)$')
    if test:
        ax_left.text(
            0.05, 0.95,
            fr'$p_{{KS}} = {evo_vs_prop_test.pvalue:.2g}$',
            transform=ax_left.transAxes,
            va="top",
            fontsize=12
        )

    # Right Panel
    counts2, bin_edges2, _ = thresholded_histogram(data=bdfe_evo, threshold=2, final_bins=12)
    anc2_counts, anc2_bin_edges, _ = thresholded_histogram(data=prop_bdfe_evo, threshold=3, final_bins=22)
    dfe2_counts, dfe2_bin_edges, _ = thresholded_histogram(data=dfe_anc, threshold=8, final_bins=24)
    bin_edges2 = bin_edges2 + xlim * shift_frac
    dfe2_bin_edges = dfe2_bin_edges - xlim * shift_frac
    anc2_bin_edges = anc2_bin_edges - xlim * shift_frac
    max1 = np.max(counts2)
    max2 = np.max(anc2_counts)
    max3 = np.max(dfe2_counts)
    ymax = max(max1, max2, max3)
    ylim = ymax * (1 + z_frac)
    z = ylim * z_frac
    counts2_shifted = counts2 + z
    ax_right.set_xlim(-xlim, xlim)
    ax_right.tick_params(labelsize=14)
    ax_right.set_ylim(0, ylim + 10)

    ax_right.stairs(
        values=counts2_shifted,
        edges=bin_edges2,
        baseline=0,
        fill=True,
        facecolor=EVO_FILL,
        edgecolor="black",
        lw=1.1,
        label="Evo."
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
        label="Anc."
    )

    ax_right.stairs(
        values=dfe2_counts,
        edges=dfe2_bin_edges,
        baseline=0,
        edgecolor=DFE_FILL,
        lw=1.1,
        label="DFE Anc."
    )

    ax_right.legend(frameon=False)
    ax_right.set_xlabel(r'Fitness effect $(\Delta)$')
    if test:
        ax_right.text(
            0.05, 0.95,
            fr'$p_{{KS}} = {anc_vs_prop_test.pvalue:.2g}$',
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


def create_segben(ax, dfe_anc, dfe_evo, lower_ben_limit, upper_ben_limit, ANC_FILL, EVO_FILL):
    # mask out non‐positive if you want
    valid_indices = np.isfinite(dfe_anc) & np.isfinite(dfe_evo)
    dfe_anc = dfe_anc[valid_indices]
    dfe_evo = dfe_evo[valid_indices]

    anc_mask = (dfe_anc > lower_ben_limit) & (dfe_anc < upper_ben_limit)
    evo_mask = (dfe_evo > lower_ben_limit) & (dfe_evo < upper_ben_limit)

    # positions
    x0, x1 = 1.0, 2.0

    # fetch the paired values
    anc_vals = dfe_anc[anc_mask]
    evo_from_anc = dfe_evo[anc_mask]

    evo_vals = dfe_evo[evo_mask]
    anc_from_evo = dfe_anc[evo_mask]

    # scatter evo→anc (reverse)
    ax.scatter(np.full_like(evo_vals, x1), evo_vals,
               color=EVO_FILL, label="Backwards")
    ax.scatter(np.full_like(evo_vals, x0), anc_from_evo,
               facecolors='none', edgecolors=EVO_FILL)

    # arrows from evo→anc
    for y1, y0 in zip(evo_vals, anc_from_evo):
        ax.add_patch(FancyArrowPatch((x1, y1), (x0, y0),
                                     arrowstyle='-|>', mutation_scale=12,
                                     color=EVO_FILL, linewidth=0.7))

    # scatter ancestor→evo
    ax.scatter(np.full_like(anc_vals, x0), anc_vals,
               color=ANC_FILL, label="Forward")
    ax.scatter(np.full_like(anc_vals, x1), evo_from_anc,
               facecolors='none', edgecolors=ANC_FILL)

    # arrows from anc→evo
    for y0, y1 in zip(anc_vals, evo_from_anc):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                     arrowstyle='-|>', mutation_scale=12,
                                     color=ANC_FILL, linewidth=0.7))

    # styling
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(('Anc.', 'Evo.'))
    ax.set_xlabel('Generations')
    ax.set_xlim(x0 - 0.2, x1 + 0.2)
    ax.set_ylabel(r'Fitness effect $(\Delta)$')
    ax.axhline(0, linestyle='--', color='black', linewidth=0.8)
    ax.tick_params(labelsize=14)
    # ax.legend(frameon=False)