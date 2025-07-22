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
    indices = np.random.choice(len(dfe), size=size, replace=False)
    return dfe[indices], indices


def compute_dfe_convergence(dfes, num_points, metric_func):
    initial_dfe, initial_indices = metric_func(dfes[0])
    indx_list = np.linspace(0, len(dfes) - 2, num_points, dtype=int)
    sampled_dfes = np.asarray(dfes)[indx_list]
    distances = np.zeros(num_points)
    for i in range(num_points):
        dfe_i = np.asarray(sampled_dfes[i], dtype=float)
        dfe_i = dfe_i / dfe_i.sum() if dfe_i.sum() > 0 else dfe_i
        metric_dfe_i = dfe_i[initial_indices] if len(initial_indices) > 0 else dfe_i
        metric_dfe_i = metric_dfe_i / metric_dfe_i.sum() if metric_dfe_i.sum() > 0 else metric_dfe_i
        distances[i] = cramervonmises_2samp(dfe_i, metric_dfe_i).statistic
    initial_distance = distances[0] if distances[0] != 0 else 1
    distances /= initial_distance
    return distances


# --- Load Data ---
res_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'FGM')
data_file_fgm = os.path.join(res_directory, 'fgm_rps100_del0.001_s00.01.pkl')
with open(data_file_fgm, 'rb') as f:
    data_fgm = pickle.load(f)

fgm_dfes = [np.asarray(entry['dfes'])[np.linspace(0, len(entry['dfes']) - 2, num_points, dtype=int)] for entry in
            data_fgm]

res_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'SK')
data_file_sk = os.path.join(res_directory, 'N4000_rho100_beta100_repeats50.pkl')
with open(data_file_sk, 'rb') as f:
    data_sk = pickle.load(f)

sk_dfes = []
for entry in data_sk:
    sampled_sigma_list = curate_sigma_list(entry['init_alpha'], entry['flip_seq'],
                                           np.linspace(0, len(entry['flip_seq']) - 2, num_points, dtype=int))
    sk_dfes.append([compute_dfe(sigma, entry['h'], entry['J']) for sigma in sampled_sigma_list])

res_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'NK')
data_file_nk = os.path.join(res_directory, 'N_2000_K_32_repeats_100.pkl')
with open(data_file_nk, 'rb') as f:
    data_nk = pickle.load(f)

nk_dfes = [np.asarray(entry['dfes'])[np.linspace(0, len(entry['dfes']) - 2, num_points, dtype=int)] for entry in
           data_nk]


# --- Compute Distances ---
def collect_model_convergence(dfes_list, metric_func):
    return np.array([compute_dfe_convergence(dfes, num_points, metric_func) for dfes in dfes_list])


fgm_ben = collect_model_convergence(fgm_dfes, compute_bdfe)
fgm_del = collect_model_convergence(fgm_dfes, compute_deleterious)

sk_ben = collect_model_convergence(sk_dfes, compute_bdfe)
sk_del = collect_model_convergence(sk_dfes, compute_deleterious)
sk_rand = collect_model_convergence(sk_dfes, lambda dfe: compute_random(dfe, len(dfe) // 2))

nk_ben = collect_model_convergence(nk_dfes, compute_bdfe)
nk_del = collect_model_convergence(nk_dfes, compute_deleterious)

# --- Plotting ---

x = np.linspace(0, 100, num_points, dtype=int)
start_frac = 0.0
start_index = int(start_frac * num_points)
x = x[start_index:]

def slice_stats(arr):
    return arr[:, start_index:].mean(axis=0), arr[:, start_index:].std(axis=0)


fgm_ben_mean, fgm_ben_std = slice_stats(fgm_ben)
sk_ben_mean, sk_ben_std = slice_stats(sk_ben)
nk_ben_mean, nk_ben_std = slice_stats(nk_ben)

fgm_del_mean, fgm_del_std = slice_stats(fgm_del)
sk_del_mean, sk_del_std = slice_stats(sk_del)
nk_del_mean, nk_del_std = slice_stats(nk_del)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles = ['Beneficial', 'Deleterious', 'Random']

# Plot beneficial
axes[0].errorbar(x, fgm_ben_mean, yerr=fgm_ben_std, fmt='-o', label='FGM', color=colors[0])
axes[0].errorbar(x, sk_ben_mean, yerr=sk_ben_std, fmt='-o', label='SK', color=colors[1])
axes[0].errorbar(x, nk_ben_mean, yerr=nk_ben_std, fmt='-o', label='NK', color=colors[2])
axes[0].set_title('Beneficial')
axes[0].set_xlabel('Evolutionary time (%)')
axes[0].set_ylabel('Normalized Distance')
axes[0].legend()

# Plot deleterious
axes[1].errorbar(x, fgm_del_mean, yerr=fgm_del_std, fmt='-o', label='FGM', color=colors[0])
axes[1].errorbar(x, sk_del_mean, yerr=sk_del_std, fmt='-o', label='SK', color=colors[1])
axes[1].errorbar(x, nk_del_mean, yerr=nk_del_std, fmt='-o', label='NK', color=colors[2])
axes[1].set_title('Deleterious')
axes[1].set_xlabel('Evolutionary time (%)')

plt.tight_layout()
plt.show(dpi=500)
