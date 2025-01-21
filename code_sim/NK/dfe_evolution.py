import matplotlib.pyplot as plt
from code.cmn import cmn_nk, cmn
import seaborn as sns
import os
from concurrent.futures import ProcessPoolExecutor

N = 700
K = [1, 2, 4, 8, 16]
times = [0.2, 0.4, 0.6, 0.8, 1.0]

output_dir = '../../plots/NK/dfe_evolution'
os.makedirs(output_dir, exist_ok=True)

def process_k(k):
    init_sigma = cmn.init_sigma(N)
    NK_init = cmn_nk.NK(N, k)
    f_off = NK_init.compute_fitness(init_sigma)
    flip_seq, NK = cmn_nk.relax_nk(init_sigma, NK_init, f_off)
    num_flips = len(flip_seq)
    flip_indices = [int(t * num_flips) for t in times]
    sigmas = cmn.curate_sigma_list(init_sigma, flip_seq, flip_indices)
    dfes = [cmn_nk.compute_dfe(sigma, NK, f_off) for sigma in sigmas]
    for i in range(len(times)):
        plt.figure()
        sns.histplot(dfes[i], bins=50, stat='density', color='grey', alpha=0.5, element='step', edgecolor='black')
        plt.title(f'Sigma at flip {flip_indices[i]}; N = {N}, K = {k}')
        plt.xlabel(r'$\Delta$')
        plt.ylabel(r'$P(\Delta)$')
        plt.savefig(os.path.join(output_dir, f'N_{N}_K_{k}_flip_{flip_indices[i]}.png'), dpi=300)
        plt.close()

with ProcessPoolExecutor() as executor:
    executor.map(process_k, K)
