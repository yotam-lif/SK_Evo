import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Parameters
N = 3 * (10 ** 3)  # Number of spins
k = 0  # Desired number of spins with negative local fields
bins = 150  # Number of bins for histograms
output_dir = 'figures'  # Directory to save figures
T = 0.003  # Temperature

# Create directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Initialize J matrix with entries drawn from a normal distribution
rng = np.random.default_rng()

# Modification: Sample only the upper triangle of J, make it symmetric, and set diagonal to zero
# Step 1: Initialize an empty (N, N) matrix
J = np.zeros((N, N))

# Step 2: Get the indices of the upper triangle of the matrix, excluding the diagonal
upper_tri_indices = np.triu_indices(N, k=1)  # k=1 excludes the diagonal

# Step 3: Sample random values for the upper triangle
J[upper_tri_indices] = rng.normal(loc=0.0, scale=1/N, size=len(upper_tri_indices[0]))

# Step 4: Make the matrix symmetric
J = J + J.T

# Step 5: Set diagonal elements to zero
np.fill_diagonal(J, 0)

# Initialize spins randomly as +1 or -1
alpha_0 = np.random.choice([-1, 1], size=N)

def local_fields(alpha):
    # Calculate local fields
    return alpha * np.dot(J, alpha)

# Modified flip function
def flip_sswm(alpha):
    # Calculate local fields
    local_flds = local_fields(alpha)
    # Find indices with negative local fields
    negative_indices = np.where(local_flds < 0)[0]
    if len(negative_indices) == 0:
        return alpha  # No negative local fields, cannot flip
    # Calculate probabilities proportional to minus the local field size
    probs = -local_flds[negative_indices]
    # Normalize the probabilities
    probs /= np.sum(probs)
    # Randomly choose one of the spins with negative local field to flip
    flip_index = np.random.choice(negative_indices, p=probs)
    alpha[flip_index] *= -1  # Flip the spin
    return alpha

def flip_glauber(alpha, beta):
    # Calculate local fields
    k_i = local_fields(alpha)
    # Calculate probabilities for flipping
    probs = (1 - np.tanh(k_i * beta)) / 2
    # choose to flip spins based on probs
    probs /= np.sum(probs)
    flip_indices = np.random.choice(np.arange(N), p=probs)
    alpha[flip_indices] *= -1
    return alpha

# Function to count number of negative local fields
def count_negative_local_fields(alpha):
    return np.sum(local_fields(alpha) < 0)

def flip_until_rank_k(alpha, k, T):
    count = 0
    alpha_local = alpha.copy()
    negative_count = count_negative_local_fields(alpha_local)
    while negative_count > k:
        # alpha_local = flip_glauber(alpha_local, 1/T)
        alpha_local = flip_sswm(alpha_local)
        count += 1
        print(f'Flip {count}, Negative Count: {negative_count}')
        negative_count = count_negative_local_fields(alpha_local)
    return alpha_local

# Step 1: Flip spins until only k remain with negative local fields
S = flip_until_rank_k(alpha_0, k, T)

# Step 2: Calculate local fields
lfs = local_fields(S)

# Step 4: Fit the histogram data to the given functions
# Define the functions to fit
def func1(x, sig_sq, b):
    return (x / sig_sq) * np.exp(-(x**2) / (2 * sig_sq)) + b


# Prepare data for fitting
hist, bin_edges = np.histogram(lfs, bins=bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Remove zero or negative bin centers (since the functions are defined for x > 0)
positive_indices = bin_centers > 0
xdata = bin_centers[positive_indices]
ydata = hist[positive_indices]

# Fit to func1
params1, _ = curve_fit(func1, xdata, ydata, p0=(N, 1/N), bounds=((0, 0), (np.inf ,1)))
# Fit to func2
# Step 5: Plot the histogram and the fitted functions
plt.figure(figsize=(10, 6))

# Plot histogram
sns.histplot(lfs, bins=bins, kde=False, stat='density', label='Data', color='gray', alpha=0.5)

# Plot fitted functions
x_fit = np.linspace(min(xdata), max(xdata), 200)
y_fit1 = func1(x_fit, *params1)

# Use LaTeX in legend labels
plt.plot(x_fit, y_fit1, label=rf'Fit 1: $a x e^{{- a x^2 / 2}} + b$' + '\n' + rf'$a={params1[0]:.2e}, b={params1[1]:.2e}$', color='blue')

plt.xlabel('Local Field Value')
plt.ylabel('Density')
plt.title(f'Histogram of Local Fields with Fitted Functions; N={N}')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'local_fields_fit.png'))
plt.show()
