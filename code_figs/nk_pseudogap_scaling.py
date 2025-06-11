import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl

# Global plot settings
glob_figsize = (8, 6)

plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 12

# Power-law model
def power_law(x, a, b, c):
    return a * x**b + c


def analyze_dfe_power_law(data_files, k_values, c_thr, num_bins=50):
    """
    Fit and plot power-law densities for multiple K values on a single linear figure.

    Parameters:
    - data_files: list of pickle files containing simulation outputs
    - k_values: corresponding K values
    - c_thr: threshold for effect sizes
    - num_bins: number of bins for histogram density estimation
    """
    if not data_files:
        print("No data files found.")
        return

    num_k = len(k_values)
    fig, ax = plt.subplots(figsize=glob_figsize)

    colors = sns.color_palette('CMRmap', n_colors=num_k)

    for idx, (path, K) in enumerate(zip(data_files, k_values)):
        # Load simulation data
        with open(path, 'rb') as f:
            data = pickle.load(f)
        # Extract final DFE values
        final_vals = []
        for rep in data:
            final_vals.extend(np.abs(rep['dfes'][-1]))
        final_vals = np.array(final_vals, dtype=float)

        # Select values below threshold
        s = final_vals[final_vals <= c_thr]
        if s.size == 0:
            print(f"No values <= {c_thr} for K={K}")
            continue

        # Histogram for density P(s)
        counts, edges = np.histogram(s, bins=num_bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        mask = counts > 0
        xdata = centers[mask]
        ydata = counts[mask]

        # Fit power-law: y = a*x^b + c
        popt, _ = curve_fit(power_law, xdata, ydata, p0=[1, 1, 0], maxfev=10000)
        slope = popt[1]

        # Plot fit curve, label with slope only
        ax.plot(xdata, power_law(xdata, *popt),
                color=colors[idx],
                label=rf'$M_{{K={K}}}={slope:.2f}$')
        ax.scatter(xdata, ydata, color=colors[idx], s=15)

    # Use scientific notation on x-axis
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # Axis labels and legend
    ax.set_xlabel('Effect size $s$')
    ax.set_ylabel('Density $P(s)$')
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Configuration
    C_THRESHOLD = 0.0015
    NUM_BINS = 10
    K_VALUES = [4, 8, 16, 32]

    # Locate data directory
    base = os.path.join(os.path.dirname(__file__), 'data', 'NK')
    if not os.path.isdir(base):
        base = os.path.join(os.path.dirname(__file__), '..', 'code_sim', 'data', 'NK')
    DATA_FILES = [os.path.join(base, f'N_2000_K_{k}_repeats_100.pkl') for k in K_VALUES]

    analyze_dfe_power_law(DATA_FILES, K_VALUES, c_thr=C_THRESHOLD, num_bins=NUM_BINS)
