import os
import matplotlib.pyplot as plt
import numpy as np
from cmn.cmn import curate_sigma_list
from cmn.cmn_sk import compute_dfe
import matplotlib as mpl
import pickle
from scipy.stats import cramervonmises_2samp
import seaborn as sns

plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

num_points = 30
colors = sns.color_palette("CMRmap", 3)
rng = np.random.default_rng(0)  # (only used if you call compute_random)

# ----- Helpers for DFE subsets -----

def compute_bdfe(dfe):
    dfe = np.asarray(dfe, dtype=float)
    mask = dfe > 0
    return dfe[mask], np.nonzero(mask)[0]

def compute_deleterious(dfe):
    dfe = np.asarray(dfe, dtype=float)
    mask = dfe < 0
    return dfe[mask], np.nonzero(mask)[0]

def compute_random(dfe, size):
    dfe = np.asarray(dfe, dtype=float)
    indices = rng.choice(len(dfe), size=size, replace=False)
    return dfe[indices], indices

# ----- Distance over time for one replicate -----

def compute_dfe_convergence(dfes, num_points, metric_func):
    """
    dfes: list/array of DFE arrays across time for ONE replicate.
    Returns a length=num_points vector of distances over time.
    """
    initial_dfe, initial_indices = metric_func(dfes[0])

    # choose time indices for sampling
    indx_list = np.linspace(0, len(dfes) - 2, num_points, dtype=int)
    sampled_dfes = np.asarray(dfes, dtype=object)[indx_list]

    distances = np.zeros(num_points, dtype=float)
    for i in range(num_points):
        dfe_i = np.asarray(sampled_dfes[i], dtype=float)

        # simple normalization to sum 1 if positive mass (as in your code)
        dfe_i = dfe_i / dfe_i.sum() if dfe_i.sum() > 0 else dfe_i

        # apply same metric subset indices as at t=0 (Option as in your current metric design)
        metric_dfe_i = dfe_i[initial_indices] if len(initial_indices) > 0 else dfe_i
        metric_dfe_i = metric_dfe_i / metric_dfe_i.sum() if metric_dfe_i.sum() > 0 else metric_dfe_i

        distances[i] = cramervonmises_2samp(dfe_i, metric_dfe_i).statistic

    return distances

# ----- Load Data -----

res_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'FGM')
data_file_fgm = os.path.join(res_directory, 'fgm_rps100_del0.001_s00.01.pkl')
with open(data_file_fgm, 'rb') as f:
    data_fgm = pickle.load(f)

# For FGM and NK, each entry contains a list of dfes over time; we sample num_points
fgm_dfes = [np.asarray(entry['dfes'], dtype=object)[np.linspace(0, len(entry['dfes']) - 2, num_points, dtype=int)]
            for entry in data_fgm]

res_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'SK')
data_file_sk = os.path.join(res_directory, 'N4000_rho100_beta100_repeats50.pkl')
with open(data_file_sk, 'rb') as f:
    data_sk = pickle.load(f)

sk_dfes = []
for entry in data_sk:
    sampled_sigma_list = curate_sigma_list(
        entry['init_alpha'],
        entry['flip_seq'],
        np.linspace(0, len(entry['flip_seq']) - 2, num_points, dtype=int)
    )
    # compute DFE at each sampled time
    sk_dfes.append([compute_dfe(sigma, entry['h'], entry['J']) for sigma in sampled_sigma_list])

res_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'NK')
data_file_nk = os.path.join(res_directory, 'N_2000_K_32_repeats_100.pkl')
with open(data_file_nk, 'rb') as f:
    data_nk = pickle.load(f)

nk_dfes = [np.asarray(entry['dfes'], dtype=object)[np.linspace(0, len(entry['dfes']) - 2, num_points, dtype=int)]
           for entry in data_nk]

# ----- Collect distances across replicates for each model -----

def collect_model_convergence(dfes_list, metric_func):
    """
    dfes_list: list of replicates; each replicate is a sequence of DFEs over time
    Returns array with shape (n_reps, num_points)
    """
    return np.array([compute_dfe_convergence(dfes, num_points, metric_func) for dfes in dfes_list], dtype=float)

fgm_ben = collect_model_convergence(fgm_dfes, compute_bdfe)
fgm_del = collect_model_convergence(fgm_dfes, compute_deleterious)

sk_ben = collect_model_convergence(sk_dfes, compute_bdfe)
sk_del = collect_model_convergence(sk_dfes, compute_deleterious)

nk_ben = collect_model_convergence(nk_dfes, compute_bdfe)
nk_del = collect_model_convergence(nk_dfes, compute_deleterious)

# ----- Option B: per-replicate normalization BEFORE aggregation -----

def per_replicate_normalized_stats(arr, start_index=0):
    """
    arr: shape (n_reps, num_points) distances for a given model/metric.
    Normalize each replicate by its own t=0 value, then take across-replicate mean/std.
    Handles zero initial values by ignoring those replicates (NaNs) in stats.
    Returns mean_norm, std_norm for the slice [start_index:].
    """
    arr = np.asarray(arr, dtype=float)

    # Safe per-replicate normalization by initial value
    init = arr[:, 0]  # shape (n_reps,)
    arr_norm = np.full_like(arr, np.nan)
    nonzero_mask = init != 0
    if np.any(nonzero_mask):
        arr_norm[nonzero_mask] = arr[nonzero_mask] / init[nonzero_mask, None]

    # Slice after optional burn-in/start fraction
    arr_norm_sliced = arr_norm[:, start_index:]

    mean_norm = np.nanmean(arr_norm_sliced, axis=0)
    std_norm = np.nanstd(arr_norm_sliced, axis=0, ddof=1)  # sample std across replicates
    return mean_norm, std_norm

# X-axis and optional burn-in
x = np.linspace(0, 100, num_points, dtype=int)
start_frac = 0.0
start_index = int(start_frac * num_points)
x = x[start_index:]

# Compute stats with per-replicate normalization
fgm_ben_mean, fgm_ben_std = per_replicate_normalized_stats(fgm_ben, start_index)
sk_ben_mean, sk_ben_std   = per_replicate_normalized_stats(sk_ben, start_index)
nk_ben_mean, nk_ben_std   = per_replicate_normalized_stats(nk_ben, start_index)

fgm_del_mean, fgm_del_std = per_replicate_normalized_stats(fgm_del, start_index)
sk_del_mean, sk_del_std   = per_replicate_normalized_stats(sk_del, start_index)
nk_del_mean, nk_del_std   = per_replicate_normalized_stats(nk_del, start_index)

# ----- Plotting -----

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Beneficial
axes[0].errorbar(x, fgm_ben_mean, yerr=fgm_ben_std, fmt='-o', label='FGM', color=colors[0])
axes[0].errorbar(x, sk_ben_mean,  yerr=sk_ben_std,  fmt='-o', label='SK',  color=colors[1])
axes[0].errorbar(x, nk_ben_mean,  yerr=nk_ben_std,  fmt='-o', label='NK',  color=colors[2])
axes[0].set_title('Beneficial')
axes[0].set_xlabel('Evolutionary time (%)')
axes[0].set_ylabel('Distance (per-replicate normalized)')
axes[0].legend()

# Deleterious
axes[1].errorbar(x, fgm_del_mean, yerr=fgm_del_std, fmt='-o', label='FGM', color=colors[0])
axes[1].errorbar(x, sk_del_mean,  yerr=sk_del_std,  fmt='-o', label='SK',  color=colors[1])
axes[1].errorbar(x, nk_del_mean,  yerr=nk_del_std,  fmt='-o', label='NK',  color=colors[2])
axes[1].set_title('Deleterious')
axes[1].set_xlabel('Evolutionary time (%)')
axes[1].legend()

plt.tight_layout()
plt.show(dpi=500)
