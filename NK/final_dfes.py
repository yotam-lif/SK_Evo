import matplotlib.pyplot as plt
from misc import cmn, cmn_nk

N = 700
K = [1, 2, 4, 6, 8, 10]

for k in K:
    init_sigma = cmn.init_sigma(N)
    NK = cmn_nk.NK(N, k)
    f_off = NK.compute_fitness(init_sigma)
    flip_seq = cmn_nk.relax_nk(init_sigma, NK, f_off)
    alpha_fin = cmn.compute_sigma_from_hist(init_sigma, flip_seq)
    dfe_fin = cmn_nk.compute_dfe(alpha_fin, NK, f_off)

    # Create a histogram of alpha_fin
    plt.hist(dfe_fin, bins=40, edgecolor='black', density=True)
    plt.title(f'Histogram of alpha_fin; N = {N}, K = {k}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
