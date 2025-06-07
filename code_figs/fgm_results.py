import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec
from code_sim.cmn.cmn_fgm import Fisher
from scipy.stats import gaussian_kde

"""
fgm_results.py — Fisher‑Geometric Model figure generator

Recreates the 6-panel layout used for NK/SK landscapes, but with the
Fisher‑Geometric Model (FGM).  Each panel mirrors its NK/SK counterpart:

A  Evolution of the full DFE during an adaptive walk
B  Final DFE for several dimensionalities *n*
C  Log‑histogram of beneficial DFEs (BD‑FE) late in the walk
D  Pairwise crossing of forward/backward BD‑FEs
E  Overlap of BD‑FE (evolved) and full DFE (ancestor)
F  Overlap of BD‑FE (ancestor) and full DFE (evolved)

All fonts, line widths, tick sizes, and spine settings match those in
*nk_results.py* and *sk_results.py* so figures sit seamlessly side‑by‑side
in the manuscript.
"""

# ──────────────────────────────────────────────────── global style ───────────────────────────────────
plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update(
    {
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    }
)
COLOR = sns.color_palette("CMRmap", 5)

# ───────────────────────────────────────────────────── helpers ───────────────────────────────────────
def propagate_forward(dfe_anc, dfe_evo):
    """Return beneficial ancestral effects and their evolved values."""
    bd = dfe_anc[dfe_anc > 0]
    idx = np.where(dfe_anc > 0)[0]
    return bd, dfe_evo[idx]


def propagate_backward(dfe_anc, dfe_evo):
    """Return beneficial evolved effects and their ancestral values."""
    bd = dfe_evo[dfe_evo > 0]
    idx = np.where(dfe_evo > 0)[0]
    return bd, dfe_anc[idx]


def _scale_x(x_ref, X_min, X_max, ref_min=-0.1, ref_max=0.1):
    """Affine map: [-0.1,0.1] ➔ [X_min,X_max] just like nk_results."""
    return X_min + ((x_ref - ref_min) / (ref_max - ref_min)) * (X_max - X_min)


def draw_custom_segments(ax, X_min, X_max, y_bottom, z_val, zorder, lw_main=1.0):
    """Grey baseline + five dashed ‘legs’ in NK/SK style using axis limits."""
    axis_x_min, axis_x_max = ax.get_xlim()

    xL = _scale_x(-0.09, axis_x_min, axis_x_max)
    xR = _scale_x(0.09, axis_x_min, axis_x_max)
    ax.plot([xL, xR], [z_val, z_val], ls="--", color="grey", lw=lw_main, zorder=zorder)

    for x0 in (-0.10, -0.05, 0.00, 0.05, 0.10):
        ax.plot(
            [
                _scale_x(x0, axis_x_min, axis_x_max),
                _scale_x(0.9 * x0, axis_x_min, axis_x_max),
            ],
            [y_bottom, z_val],
            ls="--",
            color="grey",
            lw=lw_main,
            zorder=zorder,
        )


# ───────────────────────────────────────────────────── panels A–C ─────────────────────────────────────
def panel_A(ax, reps, perc=(25, 50, 75, 100)):
    """KDE of the full DFE at *perc*% of the walk."""
    comb = [[] for _ in perc]
    for rep in reps:
        walk_length = len(rep["dfes"])
        for i, percent in enumerate(perc):
            comb[i].extend(rep["dfes"][int(percent * (walk_length - 1) / 100)])
    for i, dfe in enumerate(comb):
        dfe = np.array(dfe)
        kde = gaussian_kde(dfe, bw_method=0.4)
        x = np.linspace(dfe.min(), dfe.max(), 400)
        y = kde.evaluate(x)
        ax.plot(x, y + 0.003, lw=2, color=COLOR[i], label=f"$t={perc[i]}\\%$")
    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    ax.set_ylabel(r"$P(\Delta,t)$")
    ax.legend(frameon=False)


def panel_B(ax, final):
    """Final DFE for each *n* (negative side reflected)."""
    for i, (n, dfe) in enumerate(final.items()):
        dfe = np.asarray(dfe)
        kde = gaussian_kde(dfe, bw_method=0.3)
        x = np.linspace(dfe.min(), dfe.max(), 400)
        y = kde.evaluate(x)
        ax.plot(x, y, lw=2, color=COLOR[i], label=f"$n={n}$")
    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    ax.set_ylabel(r"$P(\Delta,t=100\%)$")
    ax.set_xlim(None, 0)
    ax.legend(frameon=False, loc="upper left")


def panel_C(ax, reps, perc=(70, 75, 80, 85)):
    """Log BD-FE histograms late in the walk, plus trend lines."""
    for i, p in enumerate(perc):
        bd_all = []
        for rep in reps:
            dfe = rep["dfes"][int(p * (len(rep["dfes"]) - 1) / 100)]
            bd_all.extend(dfe[dfe > 0])
        bins = 14 - 2 * i
        cnt, bins = np.histogram(bd_all, bins=bins, density=True)
        ctr = 0.5 * (bins[:-1] + bins[1:])
        ax.step(ctr, np.log(cnt + 1), where="mid", lw=2, color=COLOR[i], label=f"$t={p}\\%$")
        if len(ctr) >= 3:  # dashed guideline through first & third points
            x0, y0 = ctr[0], np.log(cnt[0] + 1)
            x1, y1 = ctr[2], np.log(cnt[2] + 1)
            m = (y1 - y0) / (x1 - x0)
            ax.plot(ctr, m * ctr + (y0 - m * x0), ls="--", lw=2, color=COLOR[i])
    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    ax.set_ylabel(r"$\ln(P(\Delta > 0,t))$")
    ax.legend(frameon=False, loc="upper right")


# ────────────────────────────────────────────────────── panel D ──────────────────────────────────────
def panel_D(ax, rep, anc_idx, evo_idx):
    """Forward/backward BD-FE crossings for one replicate."""
    d1, d2 = rep["dfes"][anc_idx], rep["dfes"][evo_idx]
    fwd, prop_fwd = propagate_forward(d1, d2)
    bwd, prop_bwd = propagate_backward(d1, d2)
    anc_pct, evo_pct = [int(x * 100 / (len(rep["dfes"]) - 1)) for x in (anc_idx, evo_idx)]
    alpha = 0.4
    # arrows + scatter
    for b, p in zip(fwd, prop_fwd):
        ax.add_patch(FancyArrowPatch((anc_pct, b), (evo_pct, p), arrowstyle="-|>", mutation_scale=10,
                                     color=COLOR[0], alpha=0.4, lw=0.75))
    sc_fwd = ax.scatter([anc_pct] * len(fwd), fwd, facecolors="none", edgecolor=COLOR[0], s=20, alpha=alpha)
    ax.scatter([evo_pct] * len(prop_fwd), prop_fwd, facecolors="none", edgecolor=COLOR[0], s=20, alpha=alpha)

    for b, p in zip(bwd, prop_bwd):
        ax.add_patch(FancyArrowPatch((evo_pct, b), (anc_pct, p), arrowstyle="-|>", mutation_scale=10,
                                     color=COLOR[2], alpha=0.3, lw=0.75))
    sc_bwd = ax.scatter([evo_pct] * len(bwd), bwd, facecolors="none", edgecolor=COLOR[2], s=20, alpha=alpha)
    ax.scatter([anc_pct] * len(prop_bwd), prop_bwd, facecolors="none", edgecolor=COLOR[2], s=20, alpha=alpha)

    ax.axhline(0, ls="--", lw=1.5, color="black")
    ax.set_xticks([anc_pct, evo_pct])
    ax.set_xlabel("% of walk completed")
    ax.set_ylabel(r"Fitness effect $(\Delta)$")
    ax.legend([sc_fwd, sc_bwd], ["Forwards", "Backwards"], frameon=False, loc="upper right")


# ───────────────────────────────────────────────────── panels E & F ─────────────────────────────────
def panel_EF(axL, axR, rep, anc_idx, evo_idx):
    """Overlap of BD-FEs and full DFE for ancestor (left) and evolved (right) focal points."""
    # extract DFEs
    anc_dfe = rep["dfes"][anc_idx]
    evo_dfe  = rep["dfes"][evo_idx]

    # beneficial subsets + propagated values
    anc_bdfe, anc_prop = propagate_forward(anc_dfe, evo_dfe)
    evo_bdfe, evo_prop = propagate_backward(anc_dfe, evo_dfe)

    frac_z = 0.2
    EVO_FILL = (*COLOR[1][:3], 0.75)
    ANC_FILL = (0.5, 0.5, 0.5, 0.4)
    anc_pct, evo_pct = [int(x * 100 / (len(rep["dfes"]) - 1)) for x in (anc_idx, evo_idx)]

    # ─── RIGHT PANEL (Evolved focal) ─────────────────────────────────
    # 1) evolved BD‑FE
    counts_evolved, bins_evolved = np.histogram(evo_bdfe, bins=10, density=True)
    z_r = frac_z * counts_evolved.max()
    axR.stairs(
        counts_evolved + z_r, bins_evolved, baseline=0,
        fill=True, facecolor=EVO_FILL, edgecolor="black", lw=1.1,
        label=r'$\mathcal{D}_{t_2} (t_2)$', zorder=0
    )

    # 2) ancestor-to-evolved propagated BD‑FE
    counts_ancprop, bins_ancprop = np.histogram(evo_prop, bins=16, density=True)
    axR.stairs(
        counts_ancprop, bins_ancprop, baseline=0,
        fill=True, facecolor=ANC_FILL, edgecolor="black", lw=1.1,
        label=r'$\mathcal{D}_{t_2} (t_1)$', zorder=3
    )

    # 3) full ancestral DFE outline
    counts_full_anc, bins_full_anc = np.histogram(anc_dfe, bins=16, density=True)
    axR.stairs(
        counts_full_anc, bins_full_anc, baseline=0,
        fill=False, edgecolor=COLOR[2], lw=1.1,
        label=r"$DFE(t_1)$", zorder=4
    )

    axR.set_xlabel(r"Fitness effect $(\Delta)$")
    axR.set_ylabel("Density")

    # mask rectangle and custom segments
    x_min, x_max = axR.get_xlim()
    mask = Rectangle(
        (x_min - 0.01, -0.015),
        (x_max - x_min) + 0.02,
        z_r + 0.015,
        facecolor="white", edgecolor="none", zorder=1
    )
    axR.add_patch(mask)
    draw_custom_segments(axR, x_min, x_max, -0.01, z_r, zorder=2, lw_main=1.1)
    axR.set_ylim(0, None)
    axR.legend(frameon=False, loc="upper left")


    # ─── LEFT PANEL (Ancestor focal) ─────────────────────────────────
    # 1) ancestor-to-evolved propagated BD‑FE
    counts_anc, bins_anc = np.histogram(anc_prop, bins=16, density=True)
    z_l = frac_z * counts_anc.max()
    axL.stairs(
        counts_anc + z_l, bins_anc, baseline=0,
        fill=True, facecolor=EVO_FILL, edgecolor="black", lw=1.1,
        label=r'$\mathcal{D}_{t_1} (t_2)$', zorder=0
    )

    # 2) full ancestor BD‑FE
    counts_benef, bins_benef = np.histogram(anc_bdfe, bins=10, density=True)
    axL.stairs(
        counts_benef, bins_benef, baseline=0,
        fill=True, facecolor=ANC_FILL, edgecolor="black", lw=1.1,
        label=r'$\mathcal{D}_{t_1} (t_1)$', zorder=4
    )

    # 3) full evolved DFE outline
    counts_full_evo, bins_full_evo = np.histogram(evo_dfe, bins=16, density=True)
    axL.stairs(
        counts_full_evo + z_l, bins_full_evo, baseline=0,
        fill=False, edgecolor=COLOR[2], lw=1.1,
        label=r'$DFE(t_2)$', zorder=1
    )

    axL.set_xlabel(r"Fitness effect $(\Delta)$")
    axL.set_ylabel("Density")

    x_min_L, x_max_L = axL.get_xlim()
    maskL = Rectangle(
        (x_min_L - 0.01, -0.015),
        (x_max_L - x_min_L) + 0.02,
        z_l + 0.015,
        facecolor="white", edgecolor="none", zorder=2
    )
    axL.add_patch(maskL)
    draw_custom_segments(axL, x_min_L, x_max_L, -0.01, z_l, zorder=3, lw_main=1.1)
    axL.set_ylim(0, None)
    axL.legend(frameon=False, loc="upper left")




# ───────────────────────────────────────────────────── main routine ─────────────────────────────────
if __name__ == "__main__":
    delta = 5 * 10 ** -2
    repeats = 20
    max_steps = 3000
    m = 1 * 10 ** 3
    sig_0 = 0.5

    reps = []
    for s in range(repeats):
        n=6
        model = Fisher(n=n, delta=delta, m=m, sig_0=sig_0, random_state=s, isotropic=True)
        z0 = np.random.normal(size=n, loc=0, scale=0.5)
        _, traj = model.relax(z0, max_steps=max_steps)
        reps.append({"dfes": [model.compute_dfe(z) for z in traj]})

    steps = len(reps[0]["dfes"])
    anc_idx = int(0.40 * (steps - 1))
    evo_idx = int(0.70 * (steps - 1))

    n_list = [4, 8, 16, 32]
    final = {}
    for n in n_list:
        final[n] = []
        for s in range(repeats):
            model = Fisher(n=n, delta=delta, m=m, sig_0=sig_0, random_state=s)
            z0 = np.random.normal(size=n)
            _, traj = model.relax(z0, max_steps=max_steps)
            final[n].extend(model.compute_dfe(traj[-1]))

    # Create figure and panels
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    axA = fig.add_subplot(gs[0, 0])
    panel_A(axA, reps)
    axA.text(-0.1, 1.05, "A", transform=axA.transAxes, fontsize=18, fontweight="bold")

    axB = fig.add_subplot(gs[0, 1])
    panel_B(axB, final)
    axB.text(-0.1, 1.05, "B", transform=axB.transAxes, fontsize=18, fontweight="bold")

    axC = fig.add_subplot(gs[0, 2])
    panel_C(axC, reps)
    axC.text(-0.1, 1.05, "C", transform=axC.transAxes, fontsize=18, fontweight="bold")

    axD = fig.add_subplot(gs[1, 0])
    panel_D(axD, reps[0], anc_idx, evo_idx)
    axD.text(-0.1, 1.05, "D", transform=axD.transAxes, fontsize=18, fontweight="bold")

    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])
    panel_EF(axE, axF, reps[0], anc_idx, evo_idx)
    axE.text(-0.1, 1.05, "E", transform=axE.transAxes, fontsize=18, fontweight="bold")
    axF.text(-0.1, 1.05, "F", transform=axF.transAxes, fontsize=18, fontweight="bold")

    for ax in [axA, axB, axC, axD, axE, axF]:
        ax.tick_params(width=1.5, length=6, which="major")
        ax.tick_params(width=1.5, length=3, which="minor")
        for sp in ax.spines.values():
            sp.set_linewidth(1.5)
        if ax in (axE, axF):
            ax.spines["bottom"].set_position(("outward", 10))
            ax.spines["left"].set_position(("outward", 10))
            ax.xaxis.set_ticks_position("bottom")
            ax.yaxis.set_ticks_position("left")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    output_dir = "../figs_paper"
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fgm_results.svg"), format="svg", bbox_inches="tight")