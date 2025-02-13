import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from code_sim.cmn import cmn_nk, cmn  # Ensure this import is correct for your environment

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
# ============================================


# Save the figure
output_dir = './figs_paper'
os.makedirs(output_dir, exist_ok=True)
fig_path = os.path.join(output_dir, 'nk_model_combined_dfes.svg')
plt.savefig(fig_path, format='svg', bbox_inches='tight')
plt.show()
