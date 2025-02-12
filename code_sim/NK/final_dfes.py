import matplotlib.pyplot as plt
from code_sim.cmn import cmn_nk, cmn
import seaborn as sns

N = 500
K = [2, 4, 6, 8, 10]

for k in K:
    init_sigma = cmn.init_sigma(N)
    NK = cmn_nk.NK(N, k)
    f_off = NK.compute_fitness(init_sigma)
    flip_seq, NK = cmn_nk.relax_nk(init_sigma, NK, f_off)
    alpha_fin = cmn.compute_sigma_from_hist(init_sigma, flip_seq)
    dfe_fin = cmn_nk.compute_dfe(alpha_fin, NK, f_off)

    # Create a histogram of alpha_fin
    sns.histplot(dfe_fin, bins=40, stat='density', color='grey', alpha=0.5, element='step', edgecolor='black')
    plt.title(f'Histogram of alpha_fin; N = {N}, K = {k}')
    plt.xlabel(r'$\Delta$')
    plt.ylabel(r'$P(\Delta)$')
    plt.show(dpi=300)
