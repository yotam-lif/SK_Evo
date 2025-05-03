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
n = 4
sigma = 0.05
r_list = [0.5, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0]  # r₀ values from 1 down to 0
labels = [f"$r={r}$" for r in r_list]
color = sns.color_palette("CMRmap", len(r_list))

# Create a symmetric grid around zero
x_min = -0.06
delta = np.linspace(x_min, -x_min, 3000)

plt.figure(figsize=(8, 5))
for r0, label in zip(r_list, labels):
    exp_r0_sq = np.exp(-r0 ** 2)
    # Mask to valid domain: -e^{-r0^2} <= delta <= 1 - e^{-r0^2}
    mask = (delta >= -exp_r0_sq) & (delta <= 1 - exp_r0_sq)
    delta_valid = delta[mask]

    # Map delta -> G
    g = -np.log(delta_valid + exp_r0_sq)

    # PDF of G: non-central or central chi-square scaled by sigma^2
    if r0 > 0:
        lam = (r0 / sigma) ** 2
        fG = ncx2.pdf(g / sigma ** 2, df=n, nc=lam) / sigma ** 2
    else:
        fG = chi2.pdf(g / sigma ** 2, df=n) / sigma ** 2

    # Jacobian transformation
    pdf_delta = fG / (delta_valid + exp_r0_sq)

    plt.plot(delta_valid, pdf_delta, label=label, color=color.pop(0))

plt.xlabel(r'$\delta$')
plt.ylabel(r'$P(\delta)$')
plt.title('$n=4$,$\\sigma=0.05$')
plt.xlim(x_min, -x_min)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("dfe.png", dpi=600)