import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from cmn.uncmn_dfe import gen_final_dfe, propagate_forward, propagate_backward
from cmn import cmn, cmn_sk
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib as mpl
from scipy.stats import gaussian_kde

# Define a consistent style
# plt.style.use('science')
plt.rcParams['font.family'] = 'sans-serif'
file_path = '../gen_data/SK/N4000_rho100_beta100_repeats50.pkl'
color = sns.color_palette('CMRmap', 5)
# Global font settings for labels, ticks, and legends
mpl.rcParams.update(
    {
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    }
)

with open(file_path, 'rb') as f:
    data = pickle.load(f)


# ───────────────────────────────────────────────────── helpers ───────────────────────────────────────
def _scale_x(x_ref, X_min, X_max, ref_min=-0.1, ref_max=0.1):
    return X_min + ((x_ref - ref_min) / (ref_max - ref_min)) * (X_max - X_min)


def draw_custom_segments(ax, y_bottom, z_val, zorder, lw_main=1.0):
    xmin, xmax = ax.get_xlim()
    limit = max(abs(xmin), abs(xmax))
    ax.set_xlim(-limit, limit)
    axis_x_min, axis_x_max = ax.get_xlim()

    xL = _scale_x(-0.09, axis_x_min, axis_x_max)
    xR = _scale_x(0.09, axis_x_min, axis_x_max)
    ax.plot([xL, xR], [z_val, z_val], ls="--", color="grey", lw=lw_main, zorder=zorder)

    for x0 in (-0.10, -0.05, 0.00, 0.05, 0.10):
        ax.plot(
            [
                _scale_x(x0, axis_x_min, axis_x_max),
                _scale_x(0.9 * x0, axis_x_min, axis_x_max),
            ],
            [y_bottom, z_val],
            ls="--",
            color="grey",
            lw=lw_main,
            zorder=zorder,
        )


# ───────────────────────────────────────────────────── panels A–C ─────────────────────────────────────
def create_fig_dfe_evol(ax, num_points, num_repeats):
    """
    Panel A: Evolution of DFEs over time.
    """
    # Pull out the relevant gen_data
    dfes = np.empty((num_repeats, num_points), dtype=object)
    for repeat in range(num_repeats):
        data_entry = data[repeat]
        sigma_initial = data_entry['init_alpha']
        h = data_entry['h']
        J = data_entry['J']
        flip_seq = data_entry['flip_seq']

        # Timeslices (in percent) along the adaptive walk
        percents = np.linspace(0, 100, num_points, dtype=int)
        ts = [int((len(flip_seq) - 1) * pct / 100) for pct in percents]
        sigma_list = cmn.curate_sigma_list(sigma_initial, flip_seq, ts)

        # Compute DFEs at each time slice
        for idx, sigma in enumerate(sigma_list):
            dfe = cmn_sk.compute_dfe(sigma, h, J)
            if dfes[repeat, idx] is None:
                dfes[repeat, idx] = []
            dfes[repeat, idx].extend(dfe)

    # Plot the DFEs
    for i in range(num_points):
        all_dfe = []
        for repeat in range(num_repeats):
            if dfes[repeat, i] is not None:
                all_dfe.extend(dfes[repeat, i])
        if all_dfe:
            kde = gaussian_kde(all_dfe, bw_method=0.4)
            x = np.linspace(min(all_dfe), max(all_dfe), 400)
            y = kde.evaluate(x)
            ax.plot(x, y + 0.003, lw=2, color=color[i % len(color)], label=f'$t={percents[i]}\\%$')

    ax.set_xlabel(r'Fitness effect $(\Delta)$')
    ax.set_ylabel(r'$P(\Delta, t)$')
    ax.legend(loc='upper left', frameon=False)


def create_fig_dfe_fin(ax, N, beta_arr, rho, num_repeats):
    """
    Panel B: SK-model DFE at t=100%.
    """
    for i, beta in enumerate(beta_arr):
        if beta != 1.0:
            dfe = gen_final_dfe(N, beta, rho, num_repeats)
        else:
            dfe = []
            for repeat in range(num_repeats):
                data_entry = data[repeat]
                sigma_initial = data_entry['init_alpha']
                h = data_entry['h']
                J = data_entry['J']
                flip_seq = data_entry['flip_seq']
                sigma = cmn.compute_sigma_from_hist(sigma_initial, flip_seq)
                dfe.extend(cmn_sk.compute_dfe(sigma, h, J))

        if len(dfe) == 0:
            continue  # Skip if `dfe` is empty

        if beta == 0.0:
            dfe = np.concatenate([dfe, -dfe])
            kde = gaussian_kde(dfe, bw_method=0.5)
        else:
            kde = gaussian_kde(dfe, bw_method=0.2)

        x_grid = np.linspace(min(dfe), max(dfe), 400)
        y_small = kde.evaluate(x_grid)
        if beta == 0.0:
            y_small *= 2

        ax.plot(
            x_grid, y_small,
            label=f'β={beta:.1f}',
            color=color[i % len(color)],
            lw=2.0
        )

    ax.set_xlabel(r'Fitness effect $(\Delta)$')
    ax.set_ylabel(r'$P(\Delta, t=100\%)$')
    ax.set_xlim(None, 0)
    ax.legend(loc='upper left', frameon=False)


def create_fig_bdfe_hists(ax, points_lst, num_bins, num_flips):
    """
    Plot the BD-FE histograms (log-transformed) for each walk percentage.
    """
    num = len(points_lst)
    bdfes = [[] for _ in range(num)]
    for repeat_idx in range(len(data)):
        repeat_data = data[repeat_idx]
        alphas = cmn.curate_sigma_list(repeat_data['init_alpha'], repeat_data['flip_seq'], points_lst)
        h = repeat_data['h']
        J = repeat_data['J']
        for j in range(num):
            bdfe = cmn_sk.compute_bdfe(alphas[j], h, J)[0]
            bdfes[j].extend(bdfe)

    flip_percent = (points_lst / (num_flips - 1)) * 100
    x_min_all, x_max_all = np.inf, -np.inf

    for i, bdfe in enumerate(bdfes):
        label = f'$t={flip_percent[i]:.0f}\\%$'
        bins_i = num_bins - (2 * i)
        hist, bin_edges = np.histogram(bdfe, bins=bins_i, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        x_min_all = min(x_min_all, bin_centers[0])
        x_max_all = max(x_max_all, bin_centers[-1])

        hist_log = np.log(hist + 1)
        ax.step(bin_centers, hist_log, where='mid', label=label, color=color[i % len(color)], lw=2)

        if len(bin_centers) >= 3:
            x0, y0 = bin_centers[0], hist_log[0]
            x1, y1 = bin_centers[2], hist_log[2]
            m = (y1 - y0) / (x1 - x0)
            b = y0 - m * x0
            line_vals = m * bin_centers + b
            ax.plot(bin_centers, line_vals, linestyle='--', color=color[i % len(color)], lw=2)

    y_lim = -0.03
    ax.set_xlabel(r'Fitness effect $(\Delta)$')
    ax.set_ylabel(r'$ln(P(\Delta > 0, t))$')
    ax.set_ylim(y_lim, None)
    ax.set_xlim(0, 1.5)
    ax.legend(loc='upper right', frameon=False)


# ────────────────────────────────────────────────────── panel D ──────────────────────────────────────
def create_fig_crossings(ax, flip1, flip2, repeat):
    data_entry = data[repeat]
    alpha_initial = data_entry['init_alpha']
    h = data_entry['h']
    J = data_entry['J']
    flip_seq = data_entry['flip_seq']

    bdfe1, prop_bdfe1, _ = propagate_forward(alpha_initial, h, J, flip_seq, flip1, flip2)
    bdfe2, prop_bdfe2, _ = propagate_backward(alpha_initial, h, J, flip_seq, flip1, flip2)
    num_flips = len(flip_seq)
    flip_anc_percent = int(flip1 * 100 / (num_flips - 1))
    flip_evo_percent = int(flip2 * 100 / (num_flips - 1))

    alpha = 0.4

    for b, p in zip(bdfe1, prop_bdfe1):
        ax.add_patch(FancyArrowPatch((flip_anc_percent, b), (flip_evo_percent, p), arrowstyle="-|>", mutation_scale=10,
                                     color=color[0], alpha=0.4, lw=0.75))
    sc_fwd = ax.scatter([flip_anc_percent] * len(bdfe1), bdfe1, facecolors="none", edgecolor=color[0], s=20,
                        alpha=alpha)
    ax.scatter([flip_evo_percent] * len(prop_bdfe1), prop_bdfe1, facecolors="none", edgecolor=color[0], s=20,
               alpha=alpha)

    for b, p in zip(bdfe2, prop_bdfe2):
        ax.add_patch(FancyArrowPatch((flip_evo_percent, b), (flip_anc_percent, p), arrowstyle="-|>", mutation_scale=10,
                                     color=color[2], alpha=0.3, lw=0.75))
    sc_bwd = ax.scatter([flip_evo_percent] * len(bdfe2), bdfe2, facecolors="none", edgecolor=color[2], s=20,
                        alpha=alpha)
    ax.scatter([flip_anc_percent] * len(prop_bdfe2), prop_bdfe2, facecolors="none", edgecolor=color[2], s=20,
               alpha=alpha)

    ax.axhline(0, ls="--", lw=1.5, color="black")
    ax.set_xticks([flip_anc_percent, flip_evo_percent])
    ax.set_xticklabels([r'$t_1$', r'$t_2$'])
    ax.set_xlabel('$\\%$ of walk completed')
    ax.set_ylabel(r'Fitness effect $(\Delta)$')
    ax.legend([sc_fwd, sc_bwd], ["Forwards", "Backwards"], frameon=False, loc="upper right")


# ───────────────────────────────────────────────────── panels E & F ─────────────────────────────────
def create_fig_dfes_overlap(ax_left, ax_right, flip1, flip2, repeat, color):
    # Get simulation gen_data parameters from the global gen_data variable.
    data_entry    = data[repeat]
    alpha_initial = data_entry['init_alpha']
    h             = data_entry['h']
    J             = data_entry['J']
    flip_seq      = data_entry['flip_seq']

    # Calculate percentages of walk completed like in create_fig_crossings
    num_flips         = len(flip_seq)

    fraction_z = 0.08  # vertical offset fraction
    lw_main    = 1.0

    # Propagate forward/backward.
    bdfe_anc, prop_bdfe_anc, dfe_evo = propagate_forward(
        alpha_initial, h, J, flip_seq, flip1, flip2
    )
    bdfe_evo, prop_bdfe_evo, dfe_anc = propagate_backward(
        alpha_initial, h, J, flip_seq, flip1, flip2
    )

    # --- Helper: scale x from reference range [-0.1, 0.1] to our fixed x–limits.
    def scale_x(x_ref, X_min, X_max, ref_min=-0.1, ref_max=0.1):
        return X_min + ((x_ref - ref_min) / (ref_max - ref_min)) * (X_max - X_min)

    # --- Custom segments drawing function.
    def draw_custom_segments(ax, X_min, X_max, y_bottom, z_val, lw):
        x_left_line  = scale_x(-0.09, X_min, X_max)
        x_right_line = scale_x(0.09,  X_min, X_max)
        ax.plot([x_left_line, x_right_line],
                [z_val,      z_val],
                linestyle="--", color="grey", lw=lw)
        segs_ref = [
            ((-0.05, y_bottom), ((-0.05 * 0.9), z_val)),
            (( 0.05, y_bottom), (( 0.05 * 0.9), z_val)),
            ((-0.1,  y_bottom), (-0.09,          z_val)),
            (( 0.1,  y_bottom), ( 0.09,          z_val)),
            (( 0.0,  y_bottom), ( 0.0,           z_val)),
        ]
        for (pt0, pt1) in segs_ref:
            x0 = scale_x(pt0[0], X_min, X_max)
            x1 = scale_x(pt1[0], X_min, X_max)
            ax.plot([x0, x1], [pt0[1], pt1[1]], linestyle="--", color="grey", lw=lw)

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

    # 1) propagated ancestral bDFE (shifted up by z)
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
        label=r'$\mathcal{D}_{t_1}(t_2)$',
        zorder=0
    )

    # white-out rectangle behind the overlay lines
    y_bottom_left = -0.01
    eps_height = 0.005
    eps_width  = 0.1
    rect_left = Rectangle(
        (left_X_min - eps_width, y_bottom_left - eps_height),
        (left_X_max - left_X_min) + 2 * eps_width,
        (z - y_bottom_left) + 2 * eps_height,
        facecolor="white", edgecolor="none", zorder=2
    )
    ax_left.add_patch(rect_left)

    # 2) ancestral bDFE at t1 (filled grey)
    anc_counts, anc_bin_edges = np.histogram(bdfe_anc, bins=10, density=True)
    anc_bin_edges = anc_bin_edges + 0.001  # small shift to separate outlines
    ax_left.stairs(
        values=anc_counts,
        edges=anc_bin_edges,
        baseline=0,
        fill=True,
        facecolor=ANC_FILL,
        edgecolor="black",
        lw=1.1,
        label=r'$\mathcal{D}_{t_1}(t_1)$',
        zorder=3
    )

    # 3) full evolved DFE outline (not propagated!)
    dfe_evo_counts, dfe_evo_bin_edges = np.histogram(dfe_evo, bins=20, density=True)
    dfe_evo_bin_edges = dfe_evo_bin_edges + 0.001
    ax_left.stairs(
        values=dfe_evo_counts + z - 0.01,
        edges=dfe_evo_bin_edges,
        baseline=0,
        fill=False,
        edgecolor=color[2],
        lw=1.1,
        label=r'$DFE(t_2)$',
        zorder=1
    )

    ax_left.figure.canvas.draw()
    ax_left.legend(frameon=False, loc='upper left')
    draw_custom_segments(ax_left, left_X_min, left_X_max, y_bottom_left, z, lw_main)

    # ========================
    # RIGHT PANEL
    # ========================
    right_X_min, right_X_max = -10, 10
    ax_right.cla()
    ax_right.set_xlim(right_X_min, right_X_max)
    ax_right.set_xlabel(r'Fitness effect $(\Delta)$')
    ax_right.tick_params()

    # 1) propagated evolved bDFE (filled)
    counts2, bin_edges2 = np.histogram(bdfe_evo, bins=8, density=True)
    # no large shift here; if needed use small offsets as above
    y_max_unshifted_right = ax_right.get_ylim()[1]
    z_right = fraction_z * y_max_unshifted_right
    ax_right.stairs(
        values=counts2 + z_right,
        edges=bin_edges2,
        baseline=0,
        fill=True,
        facecolor=EVO_FILL,
        edgecolor="black",
        lw=1.1,
        label=r'$\mathcal{D}_{t_2}(t_2)$'
    )

    # white-out rectangle
    rect_right = Rectangle(
        (right_X_min - eps_width, -0.01 - eps_height),
        (right_X_max - right_X_min) + 2 * eps_width,
        (z_right + eps_height),
        facecolor="white", edgecolor="none"
    )
    ax_right.add_patch(rect_right)
    draw_custom_segments(ax_right, right_X_min, right_X_max, -0.01, z_right, lw_main)

    # 2) propagated ancestral into t2 (filled grey)
    anc2_counts, anc2_bin_edges = np.histogram(prop_bdfe_evo, bins=15, density=True)
    ax_right.stairs(
        values=anc2_counts,
        edges=anc2_bin_edges,
        baseline=0,
        fill=True,
        facecolor=ANC_FILL,
        edgecolor="black",
        lw=1.1,
        label=r'$\mathcal{D}_{t_2}(t_1)$'
    )

    # 3) full ancestral DFE outline at t1
    dfe_anc2_counts, dfe_anc2_bin_edges = np.histogram(dfe_anc, bins=18, density=True)
    ax_right.stairs(
        values=dfe_anc2_counts,
        edges=dfe_anc2_bin_edges,
        baseline=0,
        fill=False,
        edgecolor=color[2],
        lw=1.1,
        label=r'$DFE(t_1)$'
    )

    ax_right.legend(frameon=False, loc='upper left')
    ax_right.set_ylim(0, None)

    # final styling
    for ax in (ax_left, ax_right):
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(width=1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

# ───────────────────────────────────────────────────── main routine ─────────────────────────────────
if __name__ == "__main__":
    num_repeats = 10
    crossings_repeat = 10
    num_flips = len(data[crossings_repeat]['flip_seq'])
    percentages_C = np.array([70, 75, 80, 85])
    percentages_D = [70, 80]
    flip_list = (percentages_C / 100 * (num_flips - 1)).astype(int)

    crossings_flip_anc = int(0.25 * (num_flips - 1))
    crossings_flip_evo = int(0.50 * (num_flips - 1))
    print(crossings_flip_evo - crossings_flip_anc)

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])

    create_fig_dfe_evol(axA, num_points=5, num_repeats=num_repeats)
    create_fig_dfe_fin(axB, N=1500, beta_arr=[0.0, 0.5, 1.0], rho=1.0, num_repeats=num_repeats)
    create_fig_bdfe_hists(axC, points_lst=flip_list, num_bins=16, num_flips=num_flips)
    create_fig_crossings(axD, crossings_flip_anc, crossings_flip_evo, crossings_repeat)
    create_fig_dfes_overlap(axE, axF, flip1=crossings_flip_anc, flip2=crossings_flip_evo, repeat=crossings_repeat, color=color)

    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    ax_list = [axA, axB, axC, axD, axE, axF]
    for i, ax in enumerate(ax_list):
        ax.text(-0.1, 1.05, panel_labels[i], transform=ax.transAxes, fontsize=18, fontweight='bold', va='bottom',
                ha='left')

    for ax in ax_list:
        ax.tick_params(width=1.5, length=6, which="major")
        ax.tick_params(width=1.5, length=3, which="minor")
        for sp in ax.spines.values():
            sp.set_linewidth(1.5)
        if ax in (axE, axF):
            ax.spines["bottom"].set_position(("outward", 10))
            ax.spines["left"].set_position(("outward", 10))
            ax.xaxis.set_ticks_position("bottom")
            ax.yaxis.set_ticks_position("left")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    output_dir = '../figs_paper'
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "sk_results.svg"), format="svg", bbox_inches='tight')

