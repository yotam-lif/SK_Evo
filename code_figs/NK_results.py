import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from code_sim.cmn import cmn_nk, cmn  # Ensure this import is correct for your environment
from code_sim.cmn import uncmn_scrambling as scr

# Configuration for the figure
plt.style.use('science')
plt.rcParams['font.family'] = 'Helvetica Neue'
sns.color_palette('CMRmap')

# File path to the NK model data
data_file = '../misc/run_data/NK/N_500_K_4_repeats_10.pkl'  # Modify this as needed

# Load the data
if os.path.exists(data_file):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
else:
    raise FileNotFoundError(f"Data file not found: {data_file}")

# Create a figure with 2 rows and 3 columns of subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# Add panel labels A, B, C, etc.
panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']
for i, ax in enumerate(axs.flat):
    ax.text(-0.1, 1.1, panel_labels[i], transform=ax.transAxes, fontsize=16, fontweight='heavy', va='top', ha='left')

# Subplot A: Histogram of combined final DFEs
# ============================================
combined_dfe = []
for entry in data:
    init_sigma = entry['init_sigma']
    flip_seq = entry['flip_seq']
    NK_model = entry['NK']

    # Compute the final configuration and DFE
    sigma_fin = cmn.compute_sigma_from_hist(init_sigma, flip_seq)
    dfe_fin = cmn_nk.compute_dfe(sigma_fin, NK_model)
    combined_dfe.extend(dfe_fin)  # Concatenate DFEs from all repeats

# Plot the histogram of the combined DFEs
sns.histplot(combined_dfe, bins=50, stat='density', color='grey', alpha=0.5, element='step', edgecolor='black',
             ax=axs[0, 0])
axs[0, 0].set_title('Histogram of Combined Final DFEs')
axs[0, 0].set_xlabel(r'$\Delta$')
axs[0, 0].set_ylabel(r'$P(\Delta)$')
# --------------------------------------------

# Subplot B: Histogram of bDFE at 80% of the adaptive walk
# ============================================
bdfe_80_percent = []
for entry in data:
    init_sigma = entry['init_sigma']
    flip_seq = entry['flip_seq']
    NK_model = entry['NK']

    # Determine the index corresponding to 80% of the flip sequence
    index_80_percent = int(0.8 * len(flip_seq))
    sigma_80_percent = cmn.compute_sigma_from_hist(init_sigma, flip_seq, index_80_percent)

    # Compute the bDFE for this configuration
    bdfe, _ = cmn_nk.compute_bdfe(sigma_80_percent, NK_model)
    bdfe_80_percent.extend(bdfe)  # Concatenate bDFEs from all repeats

# Plot the histogram of the bDFE at 80% completion
sns.histplot(bdfe_80_percent, bins=50, stat='density', color='dodgerblue', alpha=0.5, element='step', edgecolor='black',
             ax=axs[0, 1])
axs[0, 1].set_title('Histogram of bDFE at 80% of Adaptive Walk')
axs[0, 1].set_xlabel(r'$\Delta$')
axs[0, 1].set_ylabel(r'$P_+(\Delta)$')
# --------------------------------------------

# Subplot C: Plot the crossings between 15% and 45% for repeat 10
# ============================================
repeat_num = 10  # Choose repeat number 10
entry = data[repeat_num]

init_sigma = entry['init_sigma']
flip_seq = entry['flip_seq']
NK_model = entry['NK']

# Calculate indices for 15% and 45% of the adaptive walk
index_15_percent = int(0.15 * len(flip_seq))
index_45_percent = int(0.45 * len(flip_seq))

# Plot the bDFE crossings
color1 = sns.color_palette('CMRmap')[0]
color2 = sns.color_palette('CMRmap')[2]
scr.gen_crossings(axs[0, 2], init_sigma, NK_model, flip_seq, index_15_percent, index_45_percent, color1, color2)

# Customize the subplot
axs[0, 2].set_title('Crossings: 15% to 45% (Repeat 10)')
axs[0, 2].set_xlabel(r'$\%$ of walk completed')
axs[0, 2].set_ylabel(r'$\Delta$')

# Save the figure
output_dir = './figs_paper'
os.makedirs(output_dir, exist_ok=True)
fig_path = os.path.join(output_dir, 'nk_model_crossings.svg')
plt.savefig(fig_path, format='svg', bbox_inches='tight')
plt.show()
# --------------------------------------------



# Save the figure
output_dir = './figs_paper'
os.makedirs(output_dir, exist_ok=True)
fig_path = os.path.join(output_dir, 'nk_model_combined_dfes.svg')
plt.savefig(fig_path, format='svg', bbox_inches='tight')
plt.show()
