import matplotlib.pyplot as plt
from code.cmn import cmn_nk, cmn
import seaborn as sns
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

N = 700
K = [2, 4, 8, 16]
num_points = 50
times = np.linspace(0.0, 1.0, num_points, dtype=float)

output_dir = '../../plots/NK/global_epi'
os.makedirs(output_dir, exist_ok=True)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue']
plt.rcParams['font.size'] = 14

def process_k(k):
    init_sigma = cmn.init_sigma(N)
    NK_init = cmn_nk.NK(N, k)
    f_off = NK_init.compute_fitness(init_sigma)
    flip_seq, NK = cmn_nk.relax_nk(init_sigma, NK_init, f_off)
    num_flips = len(flip_seq)
    flip_indices = [int(t * num_flips) for t in times]
    sigmas = cmn.curate_sigma_list(init_sigma, flip_seq, flip_indices)
    dfes = [cmn_nk.compute_dfe(sigma, NK, f_off) for sigma in sigmas]
    bdfes = [cmn_nk.compute_bdfe(sigma, NK, f_off)[0] for sigma in sigmas]

    fits = [NK.compute_fitness(sigma) for sigma in sigmas]
    max_fit = fits[-1]
    fits = [(fit / max_fit) * 100 for fit in fits]
    ranks = [cmn_nk.compute_rank(sigma, NK, f_off) / N for sigma in sigmas]
    mean_dfes = [np.mean(dfe) for dfe in dfes]
    var_dfes = [np.var(dfe) for dfe in dfes]
    mean_bdfes = [np.mean(bdfe) for bdfe in bdfes]
    var_bdfes = [np.var(bdfe) for bdfe in bdfes]

    plt.figure()
    sns.regplot(x=fits, y=mean_dfes, marker='o', label='mean DFE', scatter_kws={'s': 50})
    sns.regplot(x=fits, y=mean_bdfes, marker='o', label='mean BDFE', scatter_kws={'s': 50})
    plt.xlabel(r'Fitness (\% from maximum reached)')
    plt.legend(fontsize=12, loc='upper right', frameon=True)
    plot_path = os.path.join(output_dir, f'global_epi_means_k={k}.png')
    plt.savefig(plot_path, format='png', bbox_inches='tight', dpi=500)
    plt.close()

    plt.figure()
    sns.regplot(x=fits, y=var_dfes, marker='o', label='var DFE', scatter_kws={'s': 50})
    sns.regplot(x=fits, y=var_bdfes, marker='o', label='var BDFE', scatter_kws={'s': 50})
    plt.xlabel(r'Fitness (\% from maximum reached)')
    plt.legend(fontsize=12, loc='upper right', frameon=True)
    plot_path = os.path.join(output_dir, f'global_epi_vars_k={k}.png')
    plt.savefig(plot_path, format='png', bbox_inches='tight', dpi=500)
    plt.close()

    plt.figure()
    sns.regplot(x=fits, y=ranks, marker='o', scatter_kws={'s': 50})
    plt.xlabel(r'Fitness (\% from maximum reached)')
    plot_path = os.path.join(output_dir, f'global_epi_ranks_k={k}.png')
    plt.savefig(plot_path, format='png', bbox_inches='tight', dpi=500)
    plt.close()

with ProcessPoolExecutor() as executor:
    executor.map(process_k, K)