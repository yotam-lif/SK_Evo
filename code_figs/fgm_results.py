import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec
from code_sim.cmn.cmn_fgm import Fisher

"""
fgm_results.py — Fisher-Geometric Model figure generator

Recreates the 6-panel layout used for NK/SK landscapes, but with the
Fisher-Geometric Model (FGM).  Each panel mirrors its NK/SK counterpart.

Panels A, C, D, E, F use a fixed dimensionality n = n_reg (default 8) —
averaged over `repeats` independent runs.  Panel B shows how the _final_
DFE changes across multiple n values (`n_list`).

A  Evolution of the full DFE during an adaptive walk (n = n_reg)
B  Final DFE for several dimensionalities *n* using Gaussian KDE
C  Log‑histogram of beneficial DFEs late in the walk (n = n_reg)
D  Pairwise crossing of forward/backward BD‑FEs (n = n_reg)
E  Overlap of BD‑FE (evolved) and full DFE (ancestor) (n = n_reg)
F  Overlap of BD‑FE (ancestor) and full DFE (evolved) (n = n_reg)

All styling, fonts, and line widths match those in *nk_results.py* and
*sk_results.py* for seamless figure panels.
"""

# ──────────────────────────────────────────────────── global style ───────────────────────────────────
plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})
COLOR = sns.color_palette("CMRmap", 5)

# ───────────────────────────────────────────────────── helpers ───────────────────────────────────────

def propagate_forward(dfe_anc, dfe_evo):
    idx = np.where(dfe_anc > 0)[0]
    return dfe_anc[idx], dfe_evo[idx]


def propagate_backward(dfe_anc, dfe_evo):
    idx = np.where(dfe_evo > 0)[0]
    return dfe_evo[idx], dfe_anc[idx]

# ───────────────────────────────────────────────────── panels A–C ─────────────────────────────────────

def panel_A(ax, reps, perc=(25, 50, 75, 100), grid_points=300):
    """Average KDE of the full DFE at *perc*% of the walk for n = n_reg."""
    for i, p in enumerate(perc):
        # Collect DFEs at time p% for each replicate
        dfes_at_p = [rep["dfes"][int(p * (len(rep["dfes"]) - 1) / 100)] for rep in reps]
        dmin = min(d.min() for d in dfes_at_p)
        dmax = max(d.max() for d in dfes_at_p)
        x = np.linspace(dmin, dmax, grid_points)
        # Evaluate KDEs and average
        ys = [gaussian_kde(d, bw_method=0.3).evaluate(x) for d in dfes_at_p]
        y_mean = np.mean(ys, axis=0)
        ax.plot(x, y_mean + 0.003, lw=2, color=COLOR[i], label=f"t={p}%")
    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    ax.set_ylabel(r"$P(\Delta,t)$")
    ax.legend(frameon=False)


def panel_B(ax, final_B, bw_method=0.2, grid_points=400):
    """Average final DFE KDE for each n in n_list (panel B only)."""
    for i, (n, dlist) in enumerate(final_B.items()):
        dmin = min(d.min() for d in dlist)
        x = np.linspace(dmin, 0.0, grid_points)
        ys = [gaussian_kde(d, bw_method=bw_method).evaluate(x) for d in dlist]
        y_mean = np.mean(ys, axis=0)
        ax.plot(x, y_mean, lw=2, color=COLOR[i], label=f"n={n}")
    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    ax.set_ylabel(r"$P(\Delta,t=100\%)$")
    ax.set_xlim(None, 0)
    ax.legend(frameon=False, loc="upper left")


def panel_C(ax, reps, perc=(70, 75, 80, 85), bins_base=12):
    """Average log BD-FE histograms late in the walk for n = n_reg."""
    for i, p in enumerate(perc):
        bd_lists = [rep["dfes"][int(p * (len(rep["dfes"]) - 1) / 100)] for rep in reps]
        positive = [d[d > 0] for d in bd_lists]
        all_bd = np.concatenate(positive)
        bins = np.histogram_bin_edges(all_bd, bins=bins_base + 3 * i)
        densities = [np.histogram(d, bins=bins, density=True)[0] for d in positive]
        density_mean = np.mean(densities, axis=0)
        ctr = 0.5 * (bins[:-1] + bins[1:])
        ax.step(ctr, np.log(density_mean + 1), where='mid', lw=2, color=COLOR[i], label=f"{p}%")
        # Trend line from first 3 points
        if len(ctr) > 2:
            x0, y0 = ctr[0], np.log(density_mean[0] + 1)
            x1, y1 = ctr[2], np.log(density_mean[2] + 1)
            m = (y1 - y0) / (x1 - x0)
            ax.plot(ctr, m * ctr + (y0 - m * x0), ls='--', lw=2, color=COLOR[i])
        else:
            ax.axhline(np.log(density_mean[0] + 1), ls='--', lw=2, color=COLOR[i])
    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    ax.set_ylabel(r"$\ln P_+(\Delta,t)$")
    ax.legend(frameon=False, loc="upper right")

# ────────────────────────────────────────────────────── panel D ──────────────────────────────────────

def panel_D(ax, rep, anc_idx, evo_idx):
    """Forward/backward BD-FE crossings for one replicate."""
    d1, d2 = rep["dfes"][anc_idx], rep["dfes"][evo_idx]
    fwd, prop_fwd = propagate_forward(d1, d2)
    bwd, prop_bwd = propagate_backward(d1, d2)
    anc_pct, evo_pct = [int(x * 100 / (len(rep["dfes"]) - 1)) for x in (anc_idx, evo_idx)]
    alpha = 0.4
    for b, p in zip(fwd, prop_fwd):
        ax.add_patch(FancyArrowPatch((anc_pct, b), (evo_pct, p), arrowstyle="-|>", mutation_scale=10, color=COLOR[0], alpha=0.4, lw=0.75))
    sc_fwd = ax.scatter([anc_pct] * len(fwd), fwd, facecolors="none", edgecolor=COLOR[0], s=20, alpha=alpha)
    ax.scatter([evo_pct] * len(prop_fwd), prop_fwd, facecolors="none", edgecolor=COLOR[0], s=20, alpha=alpha)

    for b, p in zip(bwd, prop_bwd):
        ax.add_patch(FancyArrowPatch((evo_pct, b), (anc_pct, p), arrowstyle="-|>", mutation_scale=10, color=COLOR[2], alpha=0.4, lw=0.75))
    sc_bwd = ax.scatter([evo_pct] * len(bwd), bwd, facecolors="none", edgecolor=COLOR[2], s=20, alpha=alpha)
    ax.scatter([anc_pct] * len(prop_bwd), prop_bwd, facecolors="none", edgecolor=COLOR[2], s=20, alpha=alpha)

    ax.axhline(0, ls="--", lw=1.5, color="black")
    ax.set_xticks([anc_pct, evo_pct])
    ax.set_xlabel("% of walk completed")
    ax.set_ylabel(r"Fitness effect $(\\Delta)$")
    ax.legend([sc_fwd, sc_bwd], ["Forwards", "Backwards"], frameon=False, loc="upper right")

# ───────────────────────────────────────────────────── panels E & F ─────────────────────────────────

def panel_EF(axL, axR, rep, anc_idx, evo_idx):
    """Overlap of BD-FEs and full DFE for ancestor (left) and evolved (right) focal points."""
    anc_dfe = rep["dfes"][anc_idx]
    evo_dfe = rep["dfes"][evo_idx]
    anc_bdfe, anc_prop = propagate_forward(anc_dfe, evo_dfe)
    evo_bdfe, evo_prop = propagate_backward(anc_dfe, evo_dfe)

    frac_z = 0.2
    EVO_FILL = (*COLOR[1][:3], 0.75)
    ANC_FILL = (0.5, 0.5, 0.5, 0.4)
    anc_pct, evo_pct = [int(x * 100 / (len(rep["dfes"]) - 1)) for x in (anc_idx, evo_idx)]

    # Right panel
    counts_evo, bins_evo = np.histogram(evo_bdfe, bins=10, density=True)
    z_r = frac_z * counts_evo.max()
    axR.stairs(counts_evo + z_r, bins_evo, baseline=0, fill=True, facecolor=EVO_FILL, edgecolor="black", lw=1.1, label=f"Evo(${evo_pct}\\%$)", zorder=0)
    counts_ancprop, bins_ancprop = np.histogram(evo_prop, bins=16, density=True)
    axR.stairs(counts_ancprop, bins_ancprop, baseline=0, fill=True, facecolor=ANC_FILL, edgecolor="black", lw=1.1, label=f"Anc(${anc_pct}\\%$)", zorder=3)
    counts_full_anc, bins_full_anc = np.histogram(anc_dfe, bins=16, density=True)
    axR.stairs(counts_full_anc, bins_full_anc, baseline=0, fill=False, edgecolor=COLOR[2], lw=1.1, label=f"DFE(${anc_pct}\\%$)", zorder=4)
    axR.set_xlabel(r"Fitness effect $(\\Delta)$")
    axR.set_ylabel("Density")
    draw_custom_segments(axR, *axR.get_xlim(), -0.01, z_r, zorder=2)
    axR.legend(frameon=False, loc="upper left")

    # Left panel
    counts_anc, bins_anc = np.histogram(anc_prop, bins=16, density=True)
    z_l = frac_z * counts_anc.max()
    axL.stairs(counts_anc + z_l, bins_anc, baseline=0, fill=True, facecolor=EVO_FILL, edgecolor="black", lw=1.1, label=f"Evo(${evo_pct}\\%$)", zorder=0)
    counts_benef, bins_benef = np.histogram(anc_bdfe, bins=10, density=True)
    axL.stairs(counts_benef, bins_benef, baseline=0, fill=True, facecolor=ANC_FILL, edgecolor="black", lw=1.1, label=f"Anc(${anc_pct}\\%$)", zorder=4)
    counts_full_evo, bins_full_evo = np.histogram(evo_dfe, bins=16, density=True)
    axL.stairs(counts_full_evo + z_l, bins_full_evo, baseline=0, fill=False, edgecolor=COLOR[2], lw=1.1, label=f"DFE(${evo_pct}\\%$)", zorder=1)
    axL.set_xlabel(r"Fitness effect $(\\Delta)$")
    axL.set_ylabel("Density")
    draw_custom_segments(axL, *axL.get_xlim(), -0.01, z_l, zorder=2)
    axL.legend(frameon=False, loc="upper left")


# ───────────────────────────────────────────────────── main routine ─────────────────────────────────
if __name__ == "__main__":
    # Fixed dimension for all panels except B
    n_reg = 8
    repeats = 10
    delta = 5e-2
    m = 5e3
    max_steps = 3000

    # Generate trajectories for panels A, C, D–F
    reps = []
    for s in range(repeats):
        model = Fisher(n=n_reg, delta=delta, m=m, random_state=s, isotropic=True)
        z0 = np.random.normal(size=n_reg)
        flips, traj = model.relax(z0, max_steps=max_steps)
        reps.append({"dfes": [model.compute_dfe(z) for z in traj]})

    steps = len(reps[0]["dfes"])
    anc_idx = int(0.30 * (steps - 1))
    evo_idx = int(0.70 * (steps - 1))

    # Compute final DFEs for panel B across multiple n
    n_list = [4, 8, 16, 32]
    final_B = {n: [] for n in n_list}
    for n in n_list:
        for s in range(repeats):
            model = Fisher(n=n, delta=delta, m=m, random_state=s)
            z0 = np.random.normal(size=n)
            _, traj = model.relax(z0, max_steps=max_steps)
            final_B[n].append(model.compute_dfe(traj[-1]))

    # Plot all panels
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    axA = fig.add_subplot(gs[0, 0]); panel_A(axA, reps); axA.text(-0.1,1.05,'A',transform=axA.transAxes,fontsize=18,fontweight='bold')
    axB = fig.add_subplot(gs[0, 1]); panel_B(axB, final_B); axB.text(-0.1,1.05,'B',transform=axB.transAxes,fontsize=18,fontweight='bold')
    axC = fig.add_subplot(gs[0, 2]); panel_C(axC, reps); axC.text(-0.1,1.05,'C',transform=axC.transAxes,fontsize=18,fontweight='bold')

    # Panels D–F (n = n_reg)
    axD = fig.add_subplot(gs[1, 0]); panel_D(axD, reps[0], anc_idx, evo_idx); axD.text(-0.1,1.05,'D',transform=axD.transAxes,fontsize=18,fontweight='bold')
    axE = fig.add_subplot(gs[1, 1]); axF = fig.add_subplot(gs[1, 2]); panel_EF(axE, axF, reps[0], anc_idx, evo_idx)
    axE.text(-0.1,1.05,'E',transform=axE.transAxes,fontsize=18,fontweight='bold'); axF.text(-0.1,1.05,'F',transform=axF.transAxes,fontsize=18,fontweight='bold')

    os.makedirs("../figs_paper", exist_ok=True)
    fig.savefig("../figs_paper/fgm_results.svg", format="svg", bbox_inches="tight")
