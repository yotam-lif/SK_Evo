import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from code_sim.cmn import cmn_nk, cmn  # Ensure this import is correct for your environment
from code_sim.cmn import uncmn_scrambling as scr
import scienceplots

# Configuration for the figure
plt.style.use('science')
plt.rcParams['font.family'] = 'Helvetica Neue'

# File path to the NK model data
res_directory = os.path.join(os.path.dirname(__file__), '..', 'code_sim', 'data', 'NK')
data_file_K4 = os.path.join(res_directory, 'N_2000_K_4_repeats_100.pkl')
data_file_K8 = os.path.join(res_directory, 'N_2000_K_8_repeats_100.pkl')
data_file_K16 = os.path.join(res_directory, 'N_2000_K_16_repeats_100.pkl')
data_file_K32 = os.path.join(res_directory, 'N_2000_K_32_repeats_100.pkl')

data_file_arr = [data_file_K4, data_file_K8, data_file_K16, data_file_K32]
data_arr = []
# Load the data
for data_file in data_file_arr:
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            data_arr.append(data)
    else:
        raise FileNotFoundError(f"Data file not found: {data_file}")

# Create a figure with 2 rows and 3 columns of subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
fig.subplots_adjust(hspace=0.3, wspace=0.3)
color = sns.color_palette('CMRmap', n_colors=len(data_arr))

# Add panel labels A, B, C, etc.
panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']
for i, ax in enumerate(axs.flat):
    ax.text(-0.1, 1.1, panel_labels[i], transform=ax.transAxes, fontsize=16, fontweight='heavy', va='top', ha='left')

# Subplot A: Histogram of combined final DFEs
# ============================================
for i in range(len(data_arr)):
    data = data_arr[i]
    combined_dfe = []
    for entry in data:
        init_sigma = entry['init_sigma']
        flip_seq = entry['flip_seq']
        dfes = entry['dfes']

        # Compute the final configuration and DFE
        sigma_fin = cmn.compute_sigma_from_hist(init_sigma, flip_seq)
        dfe_fin = dfes[-1]
        combined_dfe.extend(dfe_fin)  # Concatenate DFEs from all repeats

    # Plot the histogram of the combined DFEs
    sns.histplot(combined_dfe, bins=50, stat='density', element='step', edgecolor=color[i % len(color)], alpha=0.0, ax=axs[0, 0], label=f'K={2**(i+2)}')
    axs[0, 0].set_title('Final DFEs')
    axs[0, 0].set_xlabel(r'$\Delta$')
    axs[0, 0].set_ylabel(r'$P(\Delta)$')
    axs[0, 0].legend()
    # --------------------------------------------

# Subplot B: Histogram of bDFE at 80% of the adaptive walk
# ============================================
for i in range(len(data_arr)):
    data = data_arr[i]
    bdfe_80_percent = []
    for entry in data:
        init_sigma = entry['init_sigma']
        flip_seq = entry['flip_seq']
        dfes = entry['dfes']

        # Determine the index corresponding to 80% of the flip sequence
        index_80_percent = int(0.8 * len(flip_seq))
        dfe_80_percent = dfes[index_80_percent]

        # Compute the bDFE for this configuration
        bdfe, _ = cmn_nk.compute_bdfe(dfe_80_percent)
        bdfe_80_percent.extend(list(bdfe))  # Concatenate bDFEs from all repeats

    # Plot the histogram of the bDFE at 80% completion
    # sns.histplot(bdfe_80_percent, bins=50, stat='density', alpha=0.0, element='step', edgecolor=color[i % len(color)], ax=axs[0, 1], label=f'K={2**(i+2)}')

    # Compute the histogram
    counts, bin_edges = np.histogram(bdfe_80_percent, bins=50, density=True)
    log_counts = np.log(counts)
    # Plot the histogram of the log of the frequencies
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    good_indices = np.where(log_counts>1)
    log_counts = log_counts[good_indices]
    bin_centers = bin_centers[good_indices]
    axs[0, 1].step(bin_centers, log_counts, where='mid', color=color[i % len(color)], label=f'K={2 ** (i + 2)}')

    axs[0, 1].set_title('BDFE at $80\\%$ of Adaptive Walk')
    axs[0, 1].set_xlabel(r'$\Delta$')
    axs[0, 1].set_ylabel(r'$ln(P_+ (\Delta))$')
    axs[0, 1].legend()
# --------------------------------------------

# Subplot C: Plot the crossings between 15% and 45% for repeat 10
# ============================================
# data_index = 2
# repeat_num = 10  # Choose repeat number 10
# data = data_arr[data_index]
# entry = data[repeat_num]
#
# init_sigma = entry['init_sigma']
# flip_seq = entry['flip_seq']
# dfes = entry['dfes']
#
# # Calculate indices for 15% and 45% of the adaptive walk
# index_15_percent = int(0.15 * len(flip_seq))
# index_45_percent = int(0.45 * len(flip_seq))
#
# # Plot the bDFE crossings
# color1 = sns.color_palette('CMRmap')[0]
# color2 = sns.color_palette('CMRmap')[2]
# scr.gen_crossings(axs[0, 2], init_sigma, NK_model, flip_seq, index_15_percent, index_45_percent, color1, color2)
#
# # Customize the subplot
# axs[0, 2].set_title('Crossings: 15% to 45% (Repeat 10)')
# axs[0, 2].set_xlabel(r'$\%$ of walk completed')
# axs[0, 2].set_ylabel(r'$\Delta$')

# Save the figure
output_dir = '../figs_paper'
os.makedirs(output_dir, exist_ok=True)
fig_path = os.path.join(output_dir, 'nk_results.svg')
plt.savefig(fig_path, format='svg', bbox_inches='tight')
plt.show()