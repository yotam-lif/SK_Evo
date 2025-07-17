import os
import matplotlib.pyplot as plt
import numpy as np
from cmn.cmn import curate_sigma_list
from cmn.cmn_sk import compute_dfe
import matplotlib as mpl
import pickle
from scipy.stats import wasserstein_distance, ks_2samp, cramervonmises_2samp
import seaborn as sns


def compute_bdfe(dfe):
    """
    Extract beneficial fitness effects and their indices from dfe array.
    """
    dfe = np.asarray(dfe, dtype=float)
    mask = dfe > 0
    return dfe[mask], np.nonzero(mask)[0]


plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update(
    {
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    }
)
num_points = 30
colors = sns.color_palette("CMRmap", 3)

def compute_dfe_convergence(dfes, num_points):
    # Get BDFE for t=0 and the indices of all initially beneficial mutations
    bdfe, bdfe_ind = compute_bdfe(dfes[0])
    # Sample num_points evenly spaced dfes
    indx_list = np.linspace(start=0, stop=len(dfes)-2, num=num_points, dtype=int)
    sampled_dfes = np.asarray(dfes)[indx_list]
    # For each dfe, get the propagated BDFE
    sampled_prop_bdfe = [dfe[bdfe_ind] for dfe in sampled_dfes]
    distances = np.zeros(num_points)
    # Calculate the EMD between the dfes and the propagated BDFE
    # Normalize both distributions for sake of EMD calculation
    for i in range(num_points):
        dfe_i = np.asarray(sampled_dfes[i], dtype=float)
        dfe_i = dfe_i / dfe_i.sum() if dfe_i.sum() > 0 else dfe_i
        prop_bdfe_i = sampled_prop_bdfe[i]
        prop_bdfe_i = prop_bdfe_i / prop_bdfe_i.sum() if prop_bdfe_i.sum() > 0 else prop_bdfe_i
        distances[i] = cramervonmises_2samp(dfe_i, prop_bdfe_i).statistic
    return distances

# FGM Params
res_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'FGM')
data_file_fgm = os.path.join(res_directory, 'fgm_repeats1000_delta0.05.pkl')
with open(data_file_fgm, 'rb') as f:
    data_fgm = pickle.load(f)
fgm_dfes = []
fgm_zsnorms = []
for data_entry in data_fgm:
    fgm_indx_list = np.linspace(start=0, stop=len(data_entry['dfes'])-2, num=num_points, dtype=int)
    sampled_fgm_dfes = np.asarray(data_entry['dfes'])[fgm_indx_list]
    sampled_fgm_zs = np.asarray(data_entry['traj'])[fgm_indx_list]
    sampled_fgm_zsnorm = [np.linalg.norm(z) for z in sampled_fgm_zs]
    fgm_dfes.append(sampled_fgm_dfes)
    fgm_zsnorms.append(sampled_fgm_zsnorm)

fgm_distances = []

for rep in fgm_dfes:
    fgm_distances.append(compute_dfe_convergence(rep, num_points))

# Convert lists to numpy arrays for easier manipulation
fgm_distances = np.array(fgm_distances)

# Compute means and standard deviations
fgm_mean = fgm_distances.mean(axis=0)
fgm_std = fgm_distances.std(axis=0)
zs_mean = np.mean(fgm_zsnorms, axis=0)

plt.errorbar(zs_mean, fgm_mean, yerr=fgm_std, fmt='o', color=colors[0], label='FGM')
# plt.plot(zs_mean, fgm_mean, 'o', color=colors[0], label='FGM')
plt.xlabel(r'$\| z \| ^2$')
plt.ylabel('Mean Distance')
plt.ylim(0, None)
plt.show(dpi=300, bbox_inches='tight')
