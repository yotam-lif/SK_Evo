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


def analyze_dfe_power_law(data_files, k_values, lower_thr, upper_thr, num_bins=50):
    """
    Fit and plot power-law densities for multiple K values on a single linear figure.

    Parameters:
    - data_files: list of pickle files containing simulation outputs
    - k_values: corresponding K values
    - lower_thr: lower threshold for effect sizes
    - upper_thr: upper threshold for effect sizes
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

        # Select values within thresholds
        s = final_vals[(final_vals >= lower_thr) & (final_vals <= upper_thr)]
        if s.size == 0:
            print(f"No values in range [{lower_thr}, {upper_thr}] for K={K}")
            continue

        # Histogram for density P(s)
        counts, edges = np.histogram(s, bins=num_bins, density=False)
        centers = (edges[:-1] + edges[1:]) / 2
        mask = counts > 0
        xdata = centers[mask]
        ydata = counts[mask]

        # Fit power-law: y = a*x^b + c
        lower_bounds = [1e3, 0, 0]  # Minimum values
        upper_bounds = [1e11, 3, 1e4]  # Maximum values
        popt, _ = curve_fit(power_law, xdata, ydata, p0=[1e4, 1, 1e3], bounds=(lower_bounds, upper_bounds), maxfev=10000)
        A, slope, c = popt

        # Calculate R²
        y_pred = power_law(xdata, *popt)
        ss_res = np.sum((ydata - y_pred) ** 2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # Plot fit curve, label with slope only
        ax.plot(xdata, power_law(xdata, *popt),
                color=colors[idx], linewidth=2,
                label=rf'$M_{{K={K}}}={slope:.2f}, R^2={r2:.2f}$')
        ax.scatter(xdata, ydata, color=colors[idx], s=40)

    # Use scientific notation on x-axis
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # Axis labels and legend
    ax.set_xlabel('Effect size $s$')
    ax.set_ylabel('Density $P(s)$')
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()

    # Second figure: raw histograms
    fig2, ax2 = plt.subplots(figsize=glob_figsize)

    for idx, (path, K) in enumerate(zip(data_files, k_values)):
        # Load simulation data
        with open(path, 'rb') as f:
            data = pickle.load(f)
        # Extract final DFE values
        final_vals = []
        for rep in data:
            final_vals.extend(np.abs(rep['dfes'][-1]))
        final_vals = np.array(final_vals, dtype=float)
        final_vals = final_vals[final_vals <= 0.02]

        # Plot raw histogram
        ax2.hist(final_vals, bins=100, histtype='step', linewidth=2, label=f'K={K}', color=colors[idx], density=True)
    #     print the value of the counts in the left most bin for each K
        counts, edges = np.histogram(final_vals, bins=300, density=True)
        left_most_bin_count = counts[0]
        print(f'K={K}, left most bin count: {left_most_bin_count:.3e}')

    # Axis labels and legend
    ax2.set_xlabel('Effect size $s$')
    ax2.set_ylabel('Frequency')
    ax2.legend(frameon=False)
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Configuration
    UPPER_THRESHOLD = 1.2 * 1e-3
    LOWER_THRESHOLD = 1e-9
    NUM_BINS = 10
    K_VALUES = [4, 8, 16, 32]

    # Locate data directory
    base = os.path.join(os.path.dirname(__file__), 'data', 'NK')
    if not os.path.isdir(base):
        base = os.path.join(os.path.dirname(__file__), '..', 'code_sim', 'data', 'NK')
    DATA_FILES = [os.path.join(base, f'N_2000_K_{k}_repeats_100.pkl') for k in K_VALUES]

    analyze_dfe_power_law(DATA_FILES, K_VALUES, lower_thr=LOWER_THRESHOLD, upper_thr=UPPER_THRESHOLD, num_bins=NUM_BINS)