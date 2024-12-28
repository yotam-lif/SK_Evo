import matplotlib.pyplot as plt
from misc import cmn, cmn_nk
import seaborn as sns
import os
import numpy as np

N = 1000
K = [2, 4, 8, 16]
num_points = 50
times = np.linspace(0.0, 1.0, num_points, dtype=float)

output_dir = '../Plots/NK/dfe_evolution'
os.makedirs(output_dir, exist_ok=True)

for k in K:
    init_sigma = cmn.init_sigma(N)
    NK_init = cmn_nk.NK(N, k)
    f_off = NK_init.compute_fitness(init_sigma)
    flip_seq, NK = cmn_nk.relax_nk(init_sigma, NK_init, f_off)
    num_flips = len(flip_seq)
    flip_indices = [int(t*num_flips) for t in times]
    sigmas = cmn.curate_sigma_list(init_sigma, flip_seq, flip_indices)
    dfes = [cmn_nk.compute_dfe(sigma, NK, f_off) for sigma in sigmas]
    bdfes = [cmn_nk.compute_bdfe(sigma, NK, f_off) for sigma in sigmas]
    fits = [NK.compute_fitness(sigma) for sigma in sigmas]
    ranks = [cmn_nk.compute_rank(sigma, NK, f_off) / N for sigma in sigmas]

    mean_dfes = [np.mean(dfe) for dfe in dfes]
    var_dfes = [np.var(dfe) for dfe in dfes]
    mean_bdfes = [np.mean(bdfe) for bdfe in bdfes]
    var_bdfes = [np.var(bdfe) for bdfe in bdfes]
