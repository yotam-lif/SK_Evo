import matplotlib.pyplot as plt
from cmn import cmn, cmn_nk
import seaborn as sns
import os
from concurrent.futures import ProcessPoolExecutor

N = 1200
K = [2, 4, 8, 16, 32]
times = [0.4, 0.6, 0.8, 0.9]

output_dir = '../Plots/NK/bdfe_evolution'
os.makedirs(output_dir, exist_ok=True)

def process_k(k):
    init_sigma = cmn.init_sigma(N)
    NK_init = cmn_nk.NK(N, k)
    f_off = NK_init.compute_fitness(init_sigma)
    flip_seq, NK = cmn_nk.relax_nk(init_sigma, NK_init, f_off)
    num_flips = len(flip_seq)
    flip_indices = [int(t * num_flips) for t in times]
    sigmas = cmn.curate_sigma_list(init_sigma, flip_seq, flip_indices)
    bdfes = [cmn_nk.compute_bdfe(sigma, NK, f_off)[0] for sigma in sigmas]
    for i in range(len(times)):
        plt.figure()
        sns.histplot(bdfes[i], bins=50, stat='density', color='grey', alpha=0.5, element='step', edgecolor='black')
        plt.title(f'Sigma at flip {flip_indices[i]}; N = {N}, K = {k}')
        plt.xlabel(r'$\Delta$')
        plt.ylabel(r'$P(\Delta)$')
        plt.savefig(os.path.join(output_dir, f'N_{N}_K_{k}_flip_{flip_indices[i]}.png'), dpi=500)
        plt.close()

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        executor.map(process_k, K)
