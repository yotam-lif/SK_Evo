import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Global bin count for all BDFE histograms
BDFE_BINS = 15


# Ensure we run relative to this file so data paths used in the original modules still work.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import the original result modules (these keep the original rcParams/style)
import fgm_results as fgm_mod
import sk_results as sk_mod
import nk_results as nk_mod
from cmn.cmn_fgm import Fisher

percents_array = [60, 65, 70, 75]
# === EMD helper: empirical vs Exp(1) ===
def _emd_to_exp1(samples, grid_size=None):
    """
    1-Wasserstein (EMD) between empirical samples and Exp(1), using quantiles.
    'samples' are assumed to be already normalized by their mean (mean = 1).
    """
    x = np.asarray(samples)
    x = x[np.isfinite(x) & (x > 0)]
    n = x.size
    if n == 0:
        return np.nan
    x.sort()
    if grid_size is None or grid_size == n:
        u = (np.arange(1, n+1) - 0.5) / n
        q_exp1 = -np.log1p(-u)
        return np.mean(np.abs(x - q_exp1))
    else:
        u = (np.arange(1, grid_size+1) - 0.5) / grid_size
        try:
            q_emp = np.quantile(x, u, method='inverted_cdf')
        except TypeError:
            q_emp = np.quantile(x, u, interpolation='nearest')
        q_exp1 = -np.log1p(-u)
        return np.mean(np.abs(q_emp - q_exp1))
  # used to generate FGM inputs exactly like the original


# === New helpers: Exponential-fit BDFE panels (replacing the previous log-histograms) ===
import math

def _ks_exp_pvalue(samples, lam):
    '''One-sample KS test against Exp(lam). Returns (D, p_approx).'''
    if len(samples) == 0 or not np.isfinite(lam) or lam <= 0:
        return np.nan, np.nan
    x = np.sort(samples)
    n = len(x)
    # Empirical CDF at each observed x
    ecdf = (np.arange(1, n+1))/n
    # Theoretical CDF for exponential
    tcdf = 1.0 - np.exp(-lam * x)
    d_plus = np.max(ecdf - tcdf)
    d_minus = np.max(tcdf - (np.arange(0, n)/n))
    D = max(d_plus, d_minus)
    # Asymptotic p-value approximation
    en = math.sqrt(n)
    if not np.isfinite(D) or D <= 0:
        return D, 1.0
    # Kolmogorov distribution complementary CDF approximation
    s = 0.0
    for j in range(1, 101):
        s += (-1)**(j-1) * math.exp(-2 * (j**2) * (en*D)**2)
    p = max(0.0, min(1.0, 2*s))
    return D, p


def _plot_empirical_vs_exp(ax, samples, color, label_text, bins=BDFE_BINS):
    """
    Normalize raw positive BDFE samples by their mean (mean = 1), plot histogram (density=True),
    overlay Exp(1) (y = exp(-x)), and return EMD(samples_norm, Exp(1)).
    """
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples) & (samples > 0)]
    if samples.size < 3:
        return None

    mu = samples.mean()
    if not np.isfinite(mu) or mu <= 0:
        return None

    samples_norm = samples / mu  # mean-normalization

    counts, edges = np.histogram(samples_norm, bins=BDFE_BINS, density=True)
    centers = 0.5*(edges[:-1] + edges[1:])
    ax.plot(centers, counts, ls='--', lw=2, color=color)
    #
    # x_max = max(samples_norm.max(), centers[-1] if centers.size else samples_norm.max())
    # x = np.linspace(0, x_max*1.05, 400)
    # y = np.exp(-x)  # Exp(1)
    # ax.plot(x, y, lw=2, color=color, label=label_text)

    emd_val = _emd_to_exp1(samples_norm)
    return emd_val
    mu = samples.mean()
    lam = 1.0/mu if mu > 0 else np.nan
    # Empirical PDF via histogram with density=True
    counts, edges = np.histogram(samples, bins=BDFE_BINS, density=True)
    centers = 0.5*(edges[:-1] + edges[1:])
    # Draw empirical as dashed
    ax.plot(centers, counts, ls='--', lw=2, color=color)
    # Draw fitted exponential as solid
    x_max = max(samples.max(), centers[-1])
    x = np.linspace(0, x_max*1.05, 400)
    y = lam * np.exp(-lam * x)
    ax.plot(x, y, lw=2, color=color, label=label_text)
    # KS stat
    D, p = _ks_exp_pvalue(samples, lam)
    return p

def plot_bdfe_exp_fgm(ax, reps, percents=percents_array):
    '''Recompute BDFE from FGM reps and overlay exponential fits.'''
    colors = getattr(fgm_mod, 'COLOR', plt.rcParams['axes.prop_cycle'].by_key()['color'])
    # Gather total number of time steps from the first rep
    for i, p in enumerate(percents):
        bdfe = []
        for rep in reps:
            T_rep = len(rep['dfes'])
            if T_rep == 0:
                continue
            t_idx = int(p/100 * (T_rep-1))
            t_idx = max(0, min(t_idx, T_rep-1))
            dfe_t = np.asarray(rep['dfes'][t_idx])
            bdfe.extend(dfe_t[dfe_t > 0])
        label = rf'$t={p}\%,\,\mathrm{{EMD}}=\,$'  # p-value appended later
        emd_val = _plot_empirical_vs_exp(ax, bdfe, colors[i % len(colors)], label, bins=BDFE_BINS)
        if emd_val is not None:
            ax.lines[-1].set_label(rf'$t={p}\%,\,\mathrm{{EMD}}={emd_val:.2g}$')
    ax.set_xlabel(r'Fitness effect $(\Delta)$')
    ax.set_ylabel(r'$P(\Delta>0, t)$')
    ax.set_xlim(0, None)
    ax.legend(frameon=False, loc='best')

def plot_bdfe_exp_sk(ax, points_lst, num_flips):
    '''Use SK cached data to recompute BDFE and overlay exponential fits.'''
    colors = getattr(sk_mod, 'color', plt.rcParams['axes.prop_cycle'].by_key()['color'])
    num = len(points_lst)
    for i in range(num):
        bdfe = []
        for repeat_idx in range(len(sk_mod.data)):
            repeat_data = sk_mod.data[repeat_idx]
            alphas = sk_mod.cmn.curate_sigma_list(repeat_data['init_alpha'], repeat_data['flip_seq'], [points_lst[i]])
            h = repeat_data['h']; J = repeat_data['J']
            bdfe_i = sk_mod.cmn_sk.compute_bdfe(alphas[0], h, J)[0]
            bdfe.extend(bdfe_i)
        emd_val = _plot_empirical_vs_exp(ax, bdfe, colors[i % len(colors)], rf'$t={(points_lst[i]/(num_flips-1))*100:.0f}\%$', bins=BDFE_BINS)
        if emd_val is not None:
            ax.lines[-1].set_label(rf'$t={(points_lst[i]/(num_flips-1))*100:.0f}\%,\,\mathrm{{EMD}}={emd_val:.2g}$')
    ax.set_xlabel(r'Fitness effect $(\Delta)$')
    ax.set_ylabel(r'$P(\Delta>0, t)$')
    ax.set_xlim(0, None)
    ax.legend(frameon=False, loc='best')

def plot_bdfe_exp_nk(ax, percents):
    '''Use NK cached nk_data to recompute BDFE and overlay exponential fits.'''
    colors = getattr(nk_mod, 'color', plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for i, p in enumerate(percents):
        bdfe = []
        for entry in nk_mod.nk_data:
            num_flips = len(entry['flip_seq'])
            t_idx = int(p * num_flips / 100)
            dfe_t = entry['dfes'][t_idx]
            bdfe_i, _ = nk_mod.cmn_nk.compute_bdfe(dfe_t)
            bdfe.extend(bdfe_i)
        emd_val = _plot_empirical_vs_exp(ax, bdfe, colors[i % len(colors)], rf'$t={p}\%$', bins=BDFE_BINS)
        if emd_val is not None:
            ax.lines[-1].set_label(rf'$t={p}\%,\,\mathrm{{EMD}}={emd_val:.2g}$')
    ax.set_xlabel(r'Fitness effect $(\Delta)$')
    ax.set_ylabel(r'$P(\Delta>0, t)$')
    ax.set_xlim(0, None)
    ax.legend(frameon=False, loc='best')

def build_fgm_inputs():
    """
    Recreate the inputs 'reps' and 'final' for panels A–C in fgm_results.py
    using the same parameters as the original script.
    """
    # Parameters matched to fgm_results.py
    sigma = 5e-2
    repeats = 25
    max_steps = 3000
    m = 2_000

    # Panel A + C: list of replicates, each having a 'dfes' trajectory
    reps = []
    for s in range(repeats):
        n = 4
        model = Fisher(n=n, sigma=sigma, m=m, random_state=s)
        _, _, dfes = model.relax(max_steps=max_steps)
        reps.append({"dfes": dfes})

    # Panel B: final DFEs for different n values
    n_list = [4, 8, 16, 32]
    final = {n: [] for n in n_list}
    for n in n_list:
        for s in range(repeats):
            model = Fisher(n=n, sigma=sigma, m=m, random_state=s)
            _, _, dfes = model.relax(max_steps=max_steps)
            final[n].extend(dfes[-1])
    return reps, final

def build_sk_inputs():
    """
    Recreate the arguments used by sk_results.py for panels A–C.
    We use the same choices as in the original main():
      - Panel A: num_points=5, num_repeats=10
      - Panel B: N=1500, beta_arr=[0.0, 0.5, 1.0], rho=1.0, num_repeats=10
      - Panel C: percentages 70,75,80,85% of the flip sequence, num_bins = BDFE_BINS
    """
    num_repeats = 10
    # Determine number of flips from one repeat (original used repeat index 10)
    crossings_repeat = 10
    num_flips = len(sk_mod.data[crossings_repeat]['flip_seq'])
    percentages_C = np.array(percents_array)
    flip_list = (percentages_C / 100 * (num_flips - 1)).astype(int)
    return {
        "num_repeats": num_repeats,
        "N": 1500,
        "beta_arr": [0.0, 0.5, 1.0],
        "rho": 1.0,
        "flip_list": flip_list,
        "num_bins": 16,
        "num_flips": num_flips,
    }

def build_nk_inputs():
    """
    nk_results.py exposes panel functions that carry their own state from import,
    so there are no extra inputs beyond the percent choices used in the original.
    """
    return {"percents": percents_array}

def main():
    # Build inputs
    reps, final = build_fgm_inputs()
    sk_args = build_sk_inputs()
    nk_args = build_nk_inputs()

    # 3x3 figure
    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.34, wspace=0.34)

    # Row 1: FGM — panels A–C from fgm_results
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    fgm_mod.panel_A(axA, reps)                             # DFE at different times
    fgm_mod.panel_B(axB, final)                            # Final DFE vs parameter (n)
    axC.clear(); plot_bdfe_exp_fgm(axC, reps)                             # BDFE at different times

    # Row 2: SK — panels D–F corresponding to A–C of the original SK file
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])
    sk_mod.create_fig_dfe_evol(axD, num_points=5, num_repeats=sk_args["num_repeats"])
    sk_mod.create_fig_dfe_fin(axE, N=sk_args["N"], beta_arr=sk_args["beta_arr"],
                              rho=sk_args["rho"], num_repeats=sk_args["num_repeats"]*5)
    axF.clear(); plot_bdfe_exp_sk(axF, points_lst=sk_args["flip_list"], num_flips=sk_args["num_flips"])

    # Row 3: NK — panels G–I corresponding to A–C of the original NK file
    axG = fig.add_subplot(gs[2, 0])
    axH = fig.add_subplot(gs[2, 1])
    axI = fig.add_subplot(gs[2, 2])
    nk_mod.create_fig_evolution_dfe(axG)
    nk_mod.create_fig_final_dfe(axH)
    axI.clear(); plot_bdfe_exp_nk(axI, percents=nk_args["percents"])

    # Label subfigures A–I in the same way as the originals
    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    for label, ax in zip(panel_labels, [axA, axB, axC, axD, axE, axF, axG, axH, axI]):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='bottom', ha='left')

        # Keep spine widths and ticks consistent with the originals
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(width=1.5, length=6, which="major")
        ax.tick_params(width=1.5, length=3, which="minor")

    # Save SVG
    out_dir = os.path.join('..', 'figs_paper')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fig1_dfe_dynamics.svg')
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
