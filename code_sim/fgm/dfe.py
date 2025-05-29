import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2, chi2
import seaborn as sns
import matplotlib as mpl

plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 12

# Parameters
n = 6
sigma = 0.05

# First figure: Full DFEs
r_list_full = [0.5, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0]  # r₀ values from 1 down to 0
labels_full = [f"$r={r}$" for r in r_list_full]
color_full = sns.color_palette("CMRmap", len(r_list_full))

x_min = -0.06
delta = np.linspace(x_min, -x_min, 2000)

plt.figure(figsize=(8, 5))
for r0, label in zip(r_list_full, labels_full):
    exp_r0_sq = np.exp(-r0 ** 2)
    mask = (delta >= -exp_r0_sq) & (delta <= 1 - exp_r0_sq)
    delta_valid = delta[mask]

    g = -np.log(delta_valid + exp_r0_sq)

    if r0 > 0:
        lam = (r0 / sigma) ** 2
        fG = ncx2.pdf(g / sigma ** 2, df=n, nc=lam) / sigma ** 2
    else:
        fG = chi2.pdf(g / sigma ** 2, df=n) / sigma ** 2

    pdf_delta = fG / (delta_valid + exp_r0_sq)
    plt.plot(delta_valid, pdf_delta, label=label, color=color_full.pop(0))

plt.xlabel(r'$\delta$')
plt.ylabel(r'$P(\delta)$')
plt.title(f'Full DFEs (n={n}, $ \sigma = {sigma}$)')
plt.xlim(x_min, -x_min)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("dfe_full.png", dpi=600)

# Second figure: Positive parts of DFEs on semi-log scale
r_list_positive = [0.005, 0.0025, 0.001]  # Different r₀ values
labels_positive = [f"$r={r}$" for r in r_list_positive]
color_positive = sns.color_palette("viridis", len(r_list_positive))

plt.figure(figsize=(8, 5))
for r0, label in zip(r_list_positive, labels_positive):
    exp_r0_sq = np.exp(-r0 ** 2)
    delta_positive = np.linspace(0, 1-exp_r0_sq, 2000)
    g = -np.log(delta_positive + exp_r0_sq)

    if r0 > 0:
        lam = (r0 / sigma) ** 2
        fG = ncx2.pdf(g / sigma ** 2, df=n, nc=lam) / sigma ** 2
    else:
        fG = chi2.pdf(g / sigma ** 2, df=n) / sigma ** 2

    pdf_delta = fG / (delta_positive + exp_r0_sq)
    pdf_delta = np.log(pdf_delta)  # Convert to log scale

    plt.plot(delta_positive, pdf_delta, label=label, color=color_positive.pop(0))

plt.xlabel(r'$\delta$')
plt.ylabel(r'$log(P(\delta))$')
plt.title(f'Positive DFEs (n={n}, $ \sigma = {sigma}$)')
plt.legend(frameon=False)
plt.xlim(0, None)
plt.tight_layout()
plt.savefig("dfe_positive.png", dpi=600)