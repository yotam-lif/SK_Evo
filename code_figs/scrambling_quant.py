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
num_points = 51
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
data_file_fgm = os.path.join(res_directory, 'fgm_repeats1000_delta0.01.pkl')
with open(data_file_fgm, 'rb') as f:
    data_fgm = pickle.load(f)
fgm_dfes = []
for data_entry in data_fgm:
    fgm_indx_list = np.linspace(start=0, stop=len(data_entry['dfes'])-2, num=num_points, dtype=int)
    sampled_fgm_dfes = np.asarray(data_entry['dfes'])[fgm_indx_list]
    fgm_dfes.append(sampled_fgm_dfes)

# SK data
res_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'SK')
data_file_sk = os.path.join(res_directory, 'N4000_rho100_beta100_repeats50.pkl')
with open(data_file_sk, 'rb') as f:
    data_sk = pickle.load(f)
sk_dfes = []
for data_entry in data_sk:
    sigma_initial = data_entry['init_alpha']
    h = data_entry['h']
    J = data_entry['J']
    flip_seq = data_entry['flip_seq']
    sk_indx_list = np.linspace(start=0, stop=len(flip_seq)-2, num=num_points, dtype=int)
    sampled_sigma_list = curate_sigma_list(sigma_initial, flip_seq, sk_indx_list)
    sampled_sk_dfes = [compute_dfe(sigma, h, J) for sigma in sampled_sigma_list]
    sk_dfes.append(sampled_sk_dfes)

# NK data
res_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'NK')
data_file_nk = os.path.join(res_directory, 'N_2000_K_4_repeats_100.pkl')
with open(data_file_nk, 'rb') as f:
    data_nk = pickle.load(f)
nk_dfes = []
for data_entry in data_nk:
    nk_indx_list = np.linspace(start=0, stop=len(data_entry['dfes']) - 2, num=num_points, dtype=int)
    sampled_nk_dfes = np.asarray(data_entry['dfes'])[nk_indx_list]
    nk_dfes.append(sampled_nk_dfes)

len_fgm = len(fgm_dfes)
len_sk = len(sk_dfes)
len_nk = len(nk_dfes)
length = min(len_fgm, len_sk, len_nk)
fgm_distances = []
sk_distances = []
nk_distances = []
for rep in fgm_dfes:
    fgm_distances.append(compute_dfe_convergence(rep, num_points))
for rep in sk_dfes:
    sk_distances.append(compute_dfe_convergence(rep, num_points))
for rep in nk_dfes:
    nk_distances.append(compute_dfe_convergence(rep, num_points))

# Convert lists to numpy arrays for easier manipulation
fgm_distances = np.array(fgm_distances)
sk_distances = np.array(sk_distances)
nk_distances = np.array(nk_distances)

# Compute means and standard deviations
fgm_mean = fgm_distances.mean(axis=0)
fgm_std = fgm_distances.std(axis=0)
sk_mean = sk_distances.mean(axis=0)
sk_std = sk_distances.std(axis=0)
nk_mean = nk_distances.mean(axis=0)
nk_std = nk_distances.std(axis=0)

x = np.linspace(0, 100, num_points, dtype=int)

# plt.errorbar(x, fgm_mean, yerr=fgm_std, fmt='o', color=colors[0], label='FGM')
plt.plot(x, fgm_mean, 'o', color=colors[0], label='FGM')
plt.xlabel('Evolutionary time (%)')
plt.ylabel('Mean Distance')
plt.ylim(0, None)
plt.show()

# plt.errorbar(x, sk_mean, yerr=sk_std, fmt='o', color=colors[1], label='SK')
plt.plot(x, sk_mean, 'o', color=colors[1], label='SK')
plt.xlabel('Evolutionary time (%)')
plt.ylabel('Mean Distance')
plt.ylim(0, None)
plt.show()

# plt.errorbar(x, nk_mean, yerr=nk_std, fmt='o', color=colors[2], label='NK')
plt.plot(x, nk_mean, 'o', color=colors[2], label='NK')
plt.xlabel('Evolutionary time (%)')
plt.ylabel('Mean Distance')
plt.ylim(0, None)
plt.show()





