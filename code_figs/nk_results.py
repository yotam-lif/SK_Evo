import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from code_sim.cmn import cmn_nk, cmn
import matplotlib as mpl
import statsmodels.api as sm
from matplotlib.patches import FancyArrowPatch, Rectangle


# ----------------------------------------------------------------
# Utility: Reflect KDE on negative data (for final DFEs, BD-FEs, etc.)
# ----------------------------------------------------------------
def reflect_kde_neg(data, bw='scott', kernel='gau', gridsize=200):
    data_arr = np.asarray(data)
    data_arr = data_arr[data_arr <= 0]
    data_reflected = -data_arr
    data_combined = np.concatenate([data_arr, data_reflected])
    kde = sm.nonparametric.KDEUnivariate(data_combined)
    kde.fit(kernel=kernel, bw=bw, fft=False, gridsize=gridsize, cut=0, clip=(-np.inf, np.inf))
    mask = (kde.support <= 0)
    x = kde.support[mask]
    y = 2.0 * kde.density[mask]
    return x, y

# ----------------------------------------------------------------
# Utility: Compute KDE for original data (without reflecting)
# ----------------------------------------------------------------
def compute_kde_original(data, bw='scott', kernel='gau', gridsize=200):
    data_arr = np.asarray(data)
    kde = sm.nonparametric.KDEUnivariate(data_arr)
    # Use cut=3 to pad the domain a bit (as in sk_results)
    kde.fit(kernel=kernel, bw=bw, fft=False, gridsize=gridsize, cut=3)
    return kde.support, kde.density

# ----------------------------------------------------------------
# Global plot style settings
# ----------------------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 12

# ----------------------------------------------------------------
# Load NK data files
# ----------------------------------------------------------------
res_directory = os.path.join(os.path.dirname(__file__), '..', 'code_sim', 'data', 'NK')
data_file_K4 = os.path.join(res_directory, 'N_2000_K_4_repeats_100.pkl')
data_file_K8 = os.path.join(res_directory, 'N_2000_K_8_repeats_100.pkl')
data_file_K16 = os.path.join(res_directory, 'N_2000_K_16_repeats_100.pkl')
data_file_K32 = os.path.join(res_directory, 'N_2000_K_32_repeats_100.pkl')

data_files = [data_file_K4, data_file_K8, data_file_K16, data_file_K32]
K_values = [4, 8, 16, 32]
data_arr = []
for file in data_files:
    if os.path.exists(file):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            data_arr.append(data)
    else:
        raise FileNotFoundError(f"Data file not found: {file}")

# Use a 5-color palette (adjust as needed)
color = sns.color_palette('CMRmap', n_colors=5)

# For panels that require K=32 data, use the last dataset.
nk_data = data_arr[-1]

# ----------------------------------------------------------------
# Panel A: Evolution of the DFE for K=32 (averaged over all repeats)
#         (Compute KDE on the original data; no reflection)
# ----------------------------------------------------------------
def create_fig_evolution_dfe(ax):
    data = nk_data
    percents = np.linspace(0, 100, 5, dtype=int)
    combined_dfes = [[] for _ in percents]
    for repeat in data:
        flip_seq = repeat['flip_seq']
        num_flips = len(flip_seq)
        ts = [int(p * num_flips / 100) for p in percents]
        for i, t in enumerate(ts):
            dfe_t = repeat['dfes'][t]
            combined_dfes[i].extend(dfe_t)
        # Compute KDE on the original data (without reflecting)
    for i, combined_dfe in enumerate(combined_dfes):
        x_kde, y_kde = compute_kde_original(combined_dfe)
        # Add a slight vertical offset (here 0.003) as in sk_results
        ax.plot(x_kde, y_kde, color=color[i % len(color)], lw=2.0, label=f'$t={percents[i]}\\%$')
    ax.set_xlabel(r'Fitness effect ($\Delta$)')
    ax.set_ylabel(r'$P(\Delta, t)$')
    ax.legend(frameon=False)
    ax.set_ylim(None, 150)

# ----------------------------------------------------------------
# Panel B: Final DFEs for different K values (aggregated over all repeats)
# ----------------------------------------------------------------
def create_fig_final_dfe(ax):
    for i in range(len(data_arr)):
        combined_dfe = []
        for entry in data_arr[i]:
            combined_dfe.extend(entry['dfes'][-1])
        x_kde, y_kde = reflect_kde_neg(combined_dfe)
        ax.plot(x_kde, y_kde, label=f'K={int(K_values[i])}', color=color[i], lw=2.0)
    ax.set_xlabel(r'Fitness effect ($\Delta$)')
    ax.set_ylabel(r'$P(\Delta, t=\infty)$')
    ax.legend(frameon=False, loc='upper left')
    ax.set_xlim(None, 0)

# ----------------------------------------------------------------
# Panel C: BD-FEs for different K values at 80% index
# ----------------------------------------------------------------
def create_fig_bdfe_80_percent(ax):
    for i in range(len(data_arr)):
        bdfe_80 = []
        for entry in data_arr[i]:
            idx80 = int(0.8 * len(entry['flip_seq']))
            dfe_80 = entry['dfes'][idx80]
            bdfe, _ = cmn_nk.compute_bdfe(dfe_80)
            bdfe_80.extend(list(bdfe))
        bins = 15 + 3 * i
        counts, bins = np.histogram(bdfe_80, bins=bins, density=True)
        log_counts = np.log(counts)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        good = np.where(log_counts > 1)
        ax.step(bin_centers[good], log_counts[good], where='mid',
                color=color[i], label=f'K={int(K_values[i])}')
        if len(bin_centers[good]) >= 3:
            x0 = bin_centers[good][0]
            y0 = log_counts[good][0]
            x1 = bin_centers[good][2]
            y1 = log_counts[good][2]
            m = (y1 - y0) / (x1 - x0)
            line_vals = m * bin_centers[good] + (y0 - m * x0)
            ax.plot(bin_centers[good], line_vals, linestyle='--', color=color[i])
        else:
            ax.axhline(y=log_counts[good][0], linestyle='--', color=color[i])
    ax.set_xlabel(r'Fitness effect ($\Delta$)')
    ax.set_ylabel(r'$ln(P_+ (\Delta, t=80\%))$')
    ax.legend(frameon=False, loc='upper right')
    ax.set_xlim(0, 0.0062)

# ----------------------------------------------------------------
# Panel D: Crossings between two time points (using repeat 0 from K=32)
# ----------------------------------------------------------------
def create_fig_crossings_single(ax, flip1, flip2, repeat):
    data_entry = nk_data[repeat]
    dfe_anc = data_entry['dfes'][flip1]
    dfe_evo = data_entry['dfes'][flip2]
    flip_seq = data_entry['flip_seq']
    num_flips = len(flip_seq)
    bdfe1, prop_bdfe1 = cmn_nk.propagate_forward(dfe_anc, dfe_evo)
    bdfe2, prop_bdfe2 = cmn_nk.propagate_backward(dfe_anc, dfe_evo)
    color1 = color[0]
    color2  = color[1]
    flip_anc_percent = int(flip1 * 100 / num_flips)
    flip_evo_percent = int(flip2 * 100 / num_flips)
    x_left = flip_anc_percent
    x_right = flip_evo_percent
    alpha = 0.4

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

# ----------------------------------------------------------------
# Panels E–F: Overlapping DFEs (overlap of BD-FE and full DFE; using repeat 0 from K=32)
# ----------------------------------------------------------------
def create_fig_dfes_overlap(ax_left, ax_right, flip1, flip2, repeat):
    # Get simulation data parameters from the global nk_data variable.
    data_entry = nk_data[repeat]
    dfe_anc = data_entry['dfes'][flip1]
    dfe_evo = data_entry['dfes'][flip2]
    flip_seq = data_entry['flip_seq']
    num_flips = len(flip_seq)
    # Propagate forward/backward.
    bdfe_anc, prop_bdfe_anc = cmn_nk.propagate_forward(dfe_anc, dfe_evo)
    bdfe_evo, prop_bdfe_evo = cmn_nk.propagate_backward(dfe_anc, dfe_evo)
    flip_anc_percent = int(flip1 * 100 / num_flips)
    flip_evo_percent = int(flip2 * 100 / num_flips)

    fraction_z = 0.2  # Adjust as needed
    lw_main = 1.0

    # Helper function to scale x values
    def scale_x(x_ref, X_min, X_max, ref_min=-0.1, ref_max=0.1):
        return X_min + ((x_ref - ref_min) / (ref_max - ref_min)) * (X_max - X_min)

    # Custom function to draw reference segments.
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
    left_X_min, left_X_max = -0.02, 0.02
    ax_left.set_xlim(left_X_min, left_X_max)
    ax_left.set_xlabel(r'Fitness effect $(\Delta)$')
    ax_left.tick_params()

    # Compute histogram for evolved propagated BD-FE
    counts, bin_edges = np.histogram(prop_bdfe_anc, bins=12, density=True)
    max_count = np.max(counts)
    z = fraction_z * max_count  # z is now a fraction of the max histogram count

    ax_left.stairs(
        values=counts + z,
        edges=bin_edges,
        baseline=0,
        fill=True,
        facecolor=EVO_FILL,
        edgecolor="black",
        lw=1.1,
        label=f'Evolved ({flip_evo_percent}%)',
        zorder=0
    )
    y_bottom_left = -0.01
    eps_height = 0.005
    eps_width = 0.1
    from matplotlib.patches import Rectangle
    rect_left = Rectangle((left_X_min - eps_width, y_bottom_left - eps_height),
                          (left_X_max - left_X_min) + 2 * eps_width,
                          (z - y_bottom_left) + 2 * eps_height,
                          facecolor="white", edgecolor="none", zorder=2)
    ax_left.add_patch(rect_left)

    anc_counts, anc_bin_edges = np.histogram(bdfe_anc, bins=8, density=True)
    anc_bin_edges += 0.001

    dfe_evo_counts, dfe_evo_bin_edges = np.histogram(prop_bdfe_anc, bins=12, density=True)
    ax_left.stairs(
        values=dfe_evo_counts + z - 2,
        edges=dfe_evo_bin_edges + 0.0003,
        baseline=0,
        fill=False,
        edgecolor=color[2],
        lw=1.1,
        label=f'Full DFE ({flip_evo_percent}%)',
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
        label=f'Ancestor ({flip_anc_percent}%)',
        zorder=3
    )

    ax_left.figure.canvas.draw()
    ax_left.legend(frameon=False, loc='upper left', fontsize=10)
    draw_custom_segments(ax_left, left_X_min, left_X_max, y_bottom_left, z, lw_main)

    # ========================
    # RIGHT PANEL
    # ========================
    right_X_min, right_X_max = -0.02, 0.02
    ax_right.set_xlim(right_X_min, right_X_max)
    ax_right.set_xlabel(r'Fitness effect $(\Delta)$')
    ax_right.tick_params()

    counts2, bin_edges2 = np.histogram(bdfe_evo, bins=8, density=True)
    bin_edges2 += 0.5
    max_count2 = np.max(counts2)
    z_right = fraction_z * max_count2
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
        label=f'Evolved ({flip_evo_percent}%)'
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
        label=f'Ancestor ({flip_anc_percent}%)'
    )
    dfe_anc2_counts, dfe_anc2_bin_edges = np.histogram(dfe_anc, bins=18, density=True)
    ax_right.stairs(
        values=dfe_anc2_counts,
        edges=dfe_anc2_bin_edges,
        baseline=0,
        fill=False,
        edgecolor=color[2],
        lw=1.1,
        label=f'Full DFE ({flip_anc_percent}%)'
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



# ----------------------------------------------------------------
# MAIN: Create subfigures and save the figure
# ----------------------------------------------------------------
def main():
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for ax in axs.flatten():
        ax.tick_params(axis='both', which='major', length=10, width=1.5, labelsize=14)
        ax.tick_params(axis='both', which='minor', length=5, width=1.6, labelsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # Label panels A–F.
    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    for i, ax in enumerate(axs.flatten()):
        ax.text(-0.1, 1.1, panel_labels[i], transform=ax.transAxes,
                fontsize=16, fontweight='heavy', va='top', ha='left')

    # Top row:
    # create_fig_evolution_dfe(axs[0, 0])   # Panel A: Evolution of DFE (K=32, no reflection)
    # create_fig_final_dfe(axs[0, 1])         # Panel B: Final DFEs for different K values
    # create_fig_bdfe_80_percent(axs[0, 2])     # Panel C: BD-FEs for different K values at 80%

    # Bottom row:
    flip1 = int(0.15 * len(nk_data[0]['flip_seq']))
    flip2 = int(0.45 * len(nk_data[0]['flip_seq']))
    # create_fig_crossings_single(axs[1, 0], flip1, flip2, repeat=0)  # Panel D: Crossings
    create_fig_dfes_overlap(axs[1, 1], axs[1, 2], flip1, flip2, repeat=0)  # Panels E and F: Overlapping DFEs

    output_dir = os.path.join('..', 'figs_paper')
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, 'nk_results.svg')
    plt.savefig(fig_path, format='svg', bbox_inches='tight')

if __name__ == "__main__":
    main()
