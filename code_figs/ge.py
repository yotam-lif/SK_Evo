import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import code_sim.cmn.cmn as cmn_mod
import code_sim.cmn.cmn_sk as sk_mod
import code_sim.cmn.cmn_nk as nk_mod
import code_sim.cmn.cmn_fgm as fgm_mod
import os
from matplotlib.ticker import ScalarFormatter

# PARAMETERS
repeats = 1
m_points = 100
seed = 42

# Set global style
sns.set_style("white")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12
})

# Setup figure
fig, axes = plt.subplots(3, 5, figsize=(20, 12), sharex='col')
stats_names = ['Mean DFE', 'Var DFE', 'Rank', 'Mean BDFE', 'Var BDFE']

# Row 1 — SK Model
N = 1000
betas = [0.0, 0.5, 1.0]
rho = 1.0
palette = sns.color_palette('CMRmap', n_colors=len(betas))

for i, beta in enumerate(betas):
    stats_vals = {name: np.zeros(m_points) for name in stats_names}

    for r in range(repeats):
        rng = np.random.default_rng(seed + r)
        sigma0 = cmn_mod.init_sigma(N)
        h = sk_mod.init_h(N, random_state=seed + r, beta=beta)
        J = sk_mod.init_J(N, random_state=seed + r, beta=beta, rho=rho)
        flip_seq = sk_mod.relax_sk(sigma0.copy(), h, J, sswm=True)
        samples = np.linspace(0, len(flip_seq), m_points, endpoint=False, dtype=int)
        fitness = [sk_mod.compute_fit_slow(cmn_mod.compute_sigma_from_hist(sigma0, flip_seq, t), h, J) for t in samples]
        f_norm = 100 * (np.array(fitness) - np.min(fitness)) / (np.max(fitness) - np.min(fitness))

        for t_idx, t in enumerate(samples):
            sigma_t = cmn_mod.compute_sigma_from_hist(sigma0, flip_seq, t)
            dfe = sk_mod.compute_dfe(sigma_t, h, J)
            BDFE, _ = sk_mod.compute_bdfe(sigma_t, h, J)
            stats_vals['Mean DFE'][t_idx] += np.mean(dfe)
            stats_vals['Var DFE'][t_idx] += np.var(dfe)
            stats_vals['Rank'][t_idx] += np.sum(dfe > 0)
            stats_vals['Mean BDFE'][t_idx] += np.mean(BDFE) if BDFE.size > 0 else 0
            stats_vals['Var BDFE'][t_idx] += np.var(BDFE) if BDFE.size > 0 else 0

    for key in stats_vals:
        stats_vals[key] /= repeats

    for j, stat in enumerate(stats_names):
        ax = axes[0, j]
        ax.plot(f_norm, stats_vals[stat], lw=2, color=palette[i], label=f'$β={beta}$')
        if i == 0:
            ax.set_title(stat)
        if j == 0:
            ax.set_ylabel("SK")
        ax.set_xlabel("Fitness (% of max)")
        ax.legend(frameon=False)

# Row 2 — NK Model
N = 600
K_values = [4, 8, 16]
palette = sns.color_palette('CMRmap', n_colors=len(K_values))

for i, K in enumerate(K_values):
    stats_vals = {name: np.zeros(m_points) for name in stats_names}

    for r in range(repeats):
        np.random.seed(seed + r)
        nk = nk_mod.NK(N=N, K=K, mean=0.0, std=1.0, seed=seed + r)
        sigma0 = cmn_mod.init_sigma(N)
        flip_hist, dfes = nk_mod.relax_nk(sigma0.copy(), nk)
        samples = np.linspace(0, len(flip_hist), m_points, endpoint=False, dtype=int)
        fitness = [nk.compute_fitness(cmn_mod.compute_sigma_from_hist(sigma0, flip_hist, t)) for t in samples]
        f_norm = 100 * (np.array(fitness) - np.min(fitness)) / (np.max(fitness) - np.min(fitness))

        for t_idx, t in enumerate(samples):
            dfe = dfes[t]
            BDFE = dfe[dfe > 0]
            stats_vals['Mean DFE'][t_idx] += np.mean(dfe)
            stats_vals['Var DFE'][t_idx] += np.var(dfe)
            stats_vals['Rank'][t_idx] += np.sum(dfe > 0)
            stats_vals['Mean BDFE'][t_idx] += np.mean(BDFE) if BDFE.size > 0 else 0
            stats_vals['Var BDFE'][t_idx] += np.var(BDFE) if BDFE.size > 0 else 0

    for key in stats_vals:
        stats_vals[key] /= repeats

    for j, stat in enumerate(stats_names):
        ax = axes[1, j]
        ax.plot(f_norm, stats_vals[stat], lw=2, color=palette[i], label=f'$K={K}$')
        if j == 0:
            ax.set_ylabel("NK")
        ax.set_xlabel("Fitness (% of max)")
        ax.legend(frameon=False)

# Row 3 — FGM Model
n_list = [4, 8, 16]
delta = 5 * 10 ** -2
m_mut = 1000
max_steps = 1000
palette = sns.color_palette('CMRmap', n_colors=len(n_list))

for i, n in enumerate(n_list):
    stats_vals = {name: np.zeros(m_points) for name in stats_names}

    for r in range(repeats):
        model = fgm_mod.Fisher(n=n, delta=delta, m=m_mut, random_state=seed + r)
        z0 = np.random.default_rng(seed + r).normal(size=n)
        _, traj = model.relax(z0, max_steps=max_steps)
        samples = np.linspace(0, len(traj) - 1, m_points, dtype=int)
        fitness = [model.compute_fitness(traj[t]) for t in samples]
        f_norm = 100 * (np.array(fitness) - np.min(fitness)) / (np.max(fitness) - np.min(fitness))

        for t_idx, t in enumerate(samples):
            dfe = model.compute_dfe(traj[t])
            BDFE, _ = model.compute_bdfe(dfe)
            stats_vals['Mean DFE'][t_idx] += np.mean(dfe)
            stats_vals['Var DFE'][t_idx] += np.var(dfe)
            stats_vals['Rank'][t_idx] += np.sum(dfe > 0)
            stats_vals['Mean BDFE'][t_idx] += np.mean(BDFE) if BDFE.size > 0 else 0
            stats_vals['Var BDFE'][t_idx] += np.var(BDFE) if BDFE.size > 0 else 0

    for key in stats_vals:
        stats_vals[key] /= repeats

    for j, stat in enumerate(stats_names):
        ax = axes[2, j]
        ax.plot(f_norm, stats_vals[stat], lw=2, color=palette[i], label=f'$n={n}$')
        if j == 0:
            ax.set_ylabel("FGM")
        ax.set_xlabel("Fitness (% of max)")
        ax.legend(frameon=False)

# Final formatting
for ax in axes.flatten():
    ax.tick_params(which='major', length=6, width=1.5)
    ax.tick_params(which='minor', length=3, width=1.5)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

# Save figure
output_dir = '../figs_paper'
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, "ge.svg"), format="svg", bbox_inches='tight')