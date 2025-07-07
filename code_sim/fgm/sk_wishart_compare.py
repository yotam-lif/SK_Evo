import numpy as np
from code_sim.cmn import cmn, cmn_sk
import os
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def simulate_dfes(N, beta, repeats, t_percents, n_dim, case, seed=None):
    """
    Run `repeats` SK‐walks of size N, beta, for coupling `case`:
      case == 'gauss'   → J = init_J(..., rho=1.0)
      case == 'wishart' → J = init_J_wishart
    Returns a dict mapping each t% to an array of shape (repeats, N) of DFEs.
    """
    all_dfes = {t: [] for t in t_percents}

    for rep in range(repeats):
        # ensure reproducibility per‐repeat
        rep_seed = None if seed is None else seed + rep
        np.random.seed(rep_seed)

        # init spin, fields
        sigma0 = cmn.init_sigma(N)
        h      = cmn_sk.init_h(N, random_state=rep_seed, beta=beta)

        # choose coupling
        if case == 'gauss':
            J = cmn_sk.init_J(N, random_state=rep_seed, beta=beta, rho=1.0)
        else:
            J = cmn_sk.init_J_wishart(N, random_state=rep_seed, n_dim=n_dim)

        # run the adaptive walk
        flip_seq = cmn_sk.relax_sk(sigma0, h, J)
        L = len(flip_seq)

        # sample DFEs at each requested t%
        for t in t_percents:
            # index into flips: 0→initial config, L→after last flip
            idx = int(np.floor(t/100.0 * L))
            sigma_t = cmn.compute_sigma_from_hist(sigma0, flip_seq, idx)
            dfe     = cmn_sk.compute_dfe(sigma_t, h, J)
            all_dfes[t].append(dfe)

    # stack into arrays of shape (repeats, N)
    for t in t_percents:
        all_dfes[t] = np.vstack(all_dfes[t])
    return all_dfes

def plot_histograms(all_gauss, all_wishart, t_percents, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for t in t_percents:
        d_gauss = all_gauss[t].ravel()
        d_wishart = all_wishart[t].ravel()

        # KS test
        ks_stat, p_val = ks_2samp(d_gauss, d_wishart)

        plt.figure(figsize=(6, 4))

        # Gaussian SK histogram
        # counts_gauss, bins_gauss = np.histogram(d_gauss, bins=30, density=True)
        # plt.step(bins_gauss[:-1], counts_gauss, where='post', label='Gaussian SK', alpha=0.6)

        # Wishart SK histogram
        counts_wishart, bins_wishart = np.histogram(d_wishart, bins=60, density=True)
        plt.step(bins_wishart[:-1], counts_wishart, where='post', label='Wishart SK', alpha=0.6)

        plt.title(f'DFE at {t}% of walk  (KS={ks_stat:.3f}, p={p_val:.3e})')
        plt.xlabel('Δ fitness')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()

        fname = os.path.join(output_dir, f'dfe_compare_{t:03d}.png')
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Saved {fname}")

if __name__ == '__main__':
    # === Parameters to edit ===
    N = 1000  # number of spins
    beta = 1.0  # SK β (epistasis strength)
    repeats = 1  # number of independent repeats (k)
    seed = 42  # base random seed
    n=4
    outdir = 'dfe_plots'  # where to save figures
    # ==========================

    # sample at t = 0, 10, 20, ..., 100 percent of each walk
    t_percents = list(range(0, 101, 10))

    print("Simulating Gaussian SK case...")
    gauss_dfes = simulate_dfes(N, beta, repeats, t_percents, n, case='gauss', seed=seed)

    print("Simulating Wishart SK case...")
    wishart_dfes = simulate_dfes(N, beta, repeats, t_percents, n, case='wishart', seed=seed + 1)

    print("Plotting histograms & computing KS tests...")
    plot_histograms(gauss_dfes, wishart_dfes, t_percents, outdir)