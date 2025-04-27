import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv

def f_G(g, r0, sigma, n):
    """
    PDF of G = ||z0 + δ||^2, a non-central chi-square with non-centrality λ = r0^2/σ^2.
    """
    return (
        1/(2*sigma**2)
        * np.exp(-(g + r0**2)/(2*sigma**2))
        * (g/(r0**2))**(n/4 - 0.5)
        * iv(n/2 - 1, r0 * np.sqrt(g) / sigma**2)
    )

def f_U(u, r0, sigma, n):
    """
    PDF of U = exp(-G), where G = ||z0 + δ||^2.
    """
    g = -np.log(u)
    return (1/u) * f_G(g, r0, sigma, n)

def f_F(f_val, r0, sigma, n):
    """
    PDF of the unscaled fitness effect F = U - exp(-r0^2).
    """
    W0 = np.exp(-r0**2)
    return f_U(f_val + W0, r0, sigma, n)

def f_delta_prime(x, r0, sigma, n):
    """
    PDF of the scaled effect δ' = F / σ^2.
    """
    f_val = sigma**2 * x
    return sigma**2 * f_F(f_val, r0, sigma, n)

# === Parameters ===
r0 = 1.0       # background distance from optimum
n  = 6         # dimension of phenotype space
sigmas = [0.05, 0.1, 0.2]

# Range for δ'
x_min, x_max = -50, 50
x_vals = np.linspace(x_min, x_max, 1000)

# === Plot ===
plt.figure(figsize=(8, 5))
for sigma in sigmas:
    y_vals = f_delta_prime(x_vals, r0, sigma, n)
    plt.plot(x_vals, y_vals, label=f'σ = {sigma:.2f}')

plt.xlabel("Scaled fitness effect $\\delta' = F/\\sigma^2$")
plt.ylabel("Density $\\sigma^2 f_F(\\sigma^2 \\delta')$")
plt.title(f"Scaled DFE for $r_0={r0}$, $n={n}$")
plt.xlim(x_min, x_max)
plt.legend()
plt.tight_layout()
plt.show()