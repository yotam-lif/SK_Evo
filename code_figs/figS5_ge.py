import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cmn import cmn, cmn_sk as sk, cmn_nk as nk, cmn_fgm as fgm
from matplotlib.ticker import ScalarFormatter

# ---------- Project style (match fig1_dfe_dynamics & results scripts) ----------
plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})
PALETTE = sns.color_palette('CMRmap', n_colors=5)

# ---------- Parameters (kept from figS5_ge.py; change here if needed) ----------
repeats = 1
m_points = 100
seed = 42

# FGM
n_list = [4, 8, 16]
sigma = 0.05
m_mut = 10 ** 3
max_steps_fgm = 10 ** 3

# SK
N_sk = 700
betas = [0.0, 0.5, 1.0]
rho = 1.0

# NK  (Note: heavy; you can bump N_nk to 400 for full run)
N_nk = 400
K_values = [4, 8, 16]

def normalize_to_percent(x):
    x = np.asarray(x, float)
    rng = x.max() - x.min()
    return 100 * (x - x.min()) / (rng if rng > 0 else 1.0)

def sigma_at_t(s0, flips, t):
    s = s0.copy()
    for i in flips[:t]:
        s[i] = -s[i]
    return s

def main():
    # One row, three subplots: FGM, SK, NK (Mean DFE vs Fitness)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False, constrained_layout=True)
    # Use consistent line widths and axes cosmetics
    spine_lw = 1.5

    # ---------- FGM ----------
    ax = axes[0]
    colors = sns.color_palette('CMRmap', n_colors=len(n_list))
    for i, n in enumerate(n_list):
        mean_dfe_traj = np.zeros(m_points, dtype=float)
        for r in range(repeats):
            model = fgm.Fisher(n=n, sigma=sigma, m=n*m_mut, random_state=seed + r)
            flips, traj, dfes = model.relax(max_steps=max_steps_fgm)
            samples = np.linspace(0, len(traj) - 1, m_points, dtype=int)
            fitness = np.array([model.compute_fitness(traj[t]) for t in samples])
            f_norm = normalize_to_percent(fitness)
            for t_idx, t in enumerate(samples):
                dfe = model.compute_dfe(traj[t])
                mean_dfe_traj[t_idx] += float(np.mean(dfe))
        mean_dfe_traj /= repeats
        sns.regplot(x=f_norm, y=mean_dfe_traj, label=fr"$n={n}$", color=colors[i], ax=ax)
    ax.set_xlabel("Fitness (% of max)")
    ax.set_ylabel(r'$\langle \Delta \rangle$')
    ax.legend(frameon=False)

    # ---------- SK ----------
    ax = axes[1]
    colors = sns.color_palette('CMRmap', n_colors=len(betas))
    for i, beta in enumerate(betas):
        mean_dfe_traj = np.zeros(m_points, dtype=float)
        for r in range(repeats):
            sigma0 = cmn.init_sigma(N_sk)
            h = sk.init_h(N_sk, random_state=seed + r, beta=beta)
            J = sk.init_J(N_sk, random_state=seed + r, beta=beta, rho=rho)
            flip_seq = sk.relax_sk(sigma0.copy(), h, J)
            samples = np.linspace(0, len(flip_seq), m_points, endpoint=False, dtype=int)
            fitness = np.array([sk.compute_fit_slow(cmn.compute_sigma_from_hist(sigma0, flip_seq, t), h, J) for t in samples])
            f_norm = normalize_to_percent(fitness)
            for t_idx, t in enumerate(samples):
                sigma_t = cmn.compute_sigma_from_hist(sigma0, flip_seq, int(t))
                dfe = sk.compute_dfe(sigma_t, h, J)
                mean_dfe_traj[t_idx] += float(np.mean(dfe))
        mean_dfe_traj /= repeats
        sns.regplot(x=f_norm, y=mean_dfe_traj, label=fr"$\beta={beta}$", color=colors[i], ax=ax)
    ax.set_xlabel("Fitness (% of max)")
    ax.legend(frameon=False)
    ax.set_ylabel(r'$\langle \Delta \rangle$')

    # ---------- NK ----------
    ax = axes[2]
    colors = sns.color_palette('CMRmap', n_colors=len(K_values))
    for i, K in enumerate(K_values):
        mean_dfe_traj = np.zeros(m_points, dtype=float)
        for r in range(repeats):
            nk_model = nk.NK(N=N_nk, K=K, mean=0.0, std=1.0, seed=seed + r)
            sigma0 = cmn.init_sigma(N_nk)
            # Either full relax to local optimum or capped steps
            flips = []
            sigma_t = sigma0.copy()
            steps = 0
            while True:
                dfe = nk.compute_dfe(sigma_t, nk_model, 0.0)
                bdfe, b_ind = nk.compute_bdfe(dfe)
                if len(b_ind) == 0:
                    break
                # Weighted by bdfe (as in sswm_choice) but safe fallback if all zeros
                probs = bdfe / bdfe.sum() if bdfe.sum() > 0 else None
                choice = np.random.choice(b_ind, p=probs)
                flips.append(int(choice))
                sigma_t[int(choice)] = -sigma_t[int(choice)]
                steps += 1
            T = len(flips) + 1
            samples = np.linspace(0, T-1, m_points, dtype=int)
            fitness = np.array([nk_model.compute_fitness(sigma_at_t(sigma0, flips, t)) for t in samples])
            f_norm = normalize_to_percent(fitness)
            for idx, t in enumerate(samples):
                s_t = sigma_at_t(sigma0, flips, int(t))
                dfe = nk.compute_dfe(s_t, nk_model, 0.0)
                mean_dfe_traj[idx] += float(np.mean(dfe))
        mean_dfe_traj /= repeats
        sns.regplot(x=f_norm, y=mean_dfe_traj, label=fr"$K={K}$", color=colors[i], ax=ax)
    ax.set_xlabel("Fitness (% of max)")
    ax.legend(frameon=False)
    ax.set_ylabel(r'$\langle \Delta \rangle$')

    # ---------- Cosmetics to match other figures ----------
    panel_labels = ['A', 'B', 'C']
    for ax, label in zip(axes, panel_labels):
        for spine in ax.spines.values():
            spine.set_linewidth(spine_lw)
        ax.tick_params(width=spine_lw, length=6, which="major")
        ax.tick_params(width=spine_lw, length=3, which="minor")
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='bottom', ha='left')
        ticks = ax.get_xticks()
        if len(ticks) > 1:
            ax.set_xlim(-0.1, 100.1)

        # 2. Scientific notation for y-axis
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)

    # Save SVG to the same paper folder convention
    out_dir = os.path.join('..', 'figs_paper')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'figS5_ge.svg')
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
