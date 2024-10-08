import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================
# Parameters and Setup
# ==========================

# Parameters
N = 2 * (10 ** 3)  # Number of spins
rho = 1  # Fraction of non-zero J elements
beta = 0.1  # Parameter for sig_J_sq
k_final = 0  # Desired number of spins with negative local fields
bins = 75  # Number of bins for histograms
output_dir = 'figures'  # Directory to save figures
T = 0.003  # Temperature

# Define the ranks to analyze
k_values = list(range(0, 1001, 100))

# Create directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# ==========================
# Initialize J Matrix (Sparse)
# ==========================

rng = np.random.default_rng()

sig_h_sq = 1 - beta
sig_J_sq = beta / (N * rho)

# Initialize an empty (N, N) matrix
J = np.zeros((N, N))

# Get the indices of the upper triangle of the matrix, excluding the diagonal
upper_tri_indices = np.triu_indices(N, k=1)  # k=1 excludes the diagonal

# Determine the number of non-zero elements based on rho
num_possible = len(upper_tri_indices[0])
num_nonzero = int(rho * num_possible)

# Randomly choose indices to be non-zero
selected_indices = rng.choice(num_possible, size=num_nonzero, replace=False)
selected_upper = (upper_tri_indices[0][selected_indices], upper_tri_indices[1][selected_indices])

# Assign Gaussian values to the selected upper triangle indices
J[selected_upper] = rng.normal(loc=0.0, scale=np.sqrt(sig_J_sq), size=num_nonzero)

# Make the matrix symmetric
J = J + J.T

# Set diagonal elements to zero
np.fill_diagonal(J, 0)

# ==========================
# Initialize External Fields and Spins
# ==========================

# Initialize external fields h with Gaussian distribution
h = rng.normal(loc=0.0, scale=np.sqrt(sig_h_sq), size=N)

# Initialize spins randomly as +1 or -1
alpha_initial = rng.choice([-1, 1], size=N)


# ==========================
# Define Helper Functions
# ==========================

def local_fields(alpha):
    """
    Calculate local fields for each spin.

    Parameters:
        alpha (np.ndarray): Array of spin values (+1 or -1).

    Returns:
        np.ndarray: Local field values for each spin.
    """
    return alpha * (h + np.dot(J, alpha))


def flip_sswm(alpha):
    """
    Perform a single spin flip using the Sequential Single Spin Flip Monte Carlo (SSWM) method.

    Parameters:
        alpha (np.ndarray): Current spin configuration.

    Returns:
        tuple: Updated spin configuration and a boolean indicating if a flip was performed.
    """
    # Calculate local fields
    local_flds = local_fields(alpha)

    # Find indices with negative local fields
    negative_indices = np.where(local_flds < 0)[0]

    if len(negative_indices) == 0:
        return alpha, False  # No flip performed

    # Calculate probabilities proportional to the magnitude of the negative local fields
    probs = -local_flds[negative_indices]

    # Normalize the probabilities
    probs /= np.sum(probs)

    # Randomly choose one of the spins with negative local field to flip
    flip_index = rng.choice(negative_indices, p=probs)

    # Flip the selected spin
    alpha[flip_index] *= -1

    return alpha, True


def count_negative_local_fields(alpha):
    """
    Count the number of spins with negative local fields.

    Parameters:
        alpha (np.ndarray): Current spin configuration.

    Returns:
        int: Number of spins with negative local fields.
    """
    return np.sum(local_fields(alpha) < 0)


def flip_until_rank_k(alpha, k):
    """
    Flip spins until only k spins have negative local fields.

    Parameters:
        alpha (np.ndarray): Initial spin configuration.
        k (int): Desired number of spins with negative local fields.

    Returns:
        tuple: Updated spin configuration and the number of flips performed.
    """
    count = 0
    alpha_local = alpha.copy()
    negative_count = count_negative_local_fields(alpha_local)

    while negative_count > k:
        alpha_local, flipped = flip_sswm(alpha_local)
        if not flipped:
            break  # No more flips possible
        count += 1
        if count % 1000 == 0:
            print(f'Flip {count}, Negative Count: {negative_count}')
        negative_count = count_negative_local_fields(alpha_local)

    return alpha_local, count


# ==========================
# Define Fitting Function
# ==========================

def func1(x, a, b):
    """
    Define the fitting function: a * x * exp(-a * x^2 / 2) + b

    Parameters:
        x (np.ndarray): Independent variable.
        a (float): Parameter a.
        b (float): Parameter b.

    Returns:
        np.ndarray: Function values.
    """
    return a * x * np.exp(-a * x ** 2 / 2) + b


# ==========================
# Process Multiple Ranks and Plot
# ==========================

# Dictionary to store results for each rank
results = {}

for k in k_values:
    print(f'\nProcessing rank k={k}...')

    # Initialize spins randomly for each rank
    alpha = rng.choice([-1, 1], size=N)

    # Flip until desired rank
    S, num_flips = flip_until_rank_k(alpha, k)
    print(f'Completed flipping: {num_flips} flips to reach k={k}')

    # Calculate local fields
    lfs = local_fields(S)

    # Store results
    results[k] = {
        'spins': S,
        'local_fields': lfs,
        'num_flips': num_flips
    }

    # Plot histogram for the current rank
    plt.figure(figsize=(10, 6))
    sns.histplot(lfs, bins=bins, kde=False, stat='density', label=f'k={k}', color='blue', alpha=0.6)
    plt.xlabel('Local Field Value')
    plt.ylabel('Density')
    plt.title(f'Histogram of Local Fields for k={k}; N={N}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'local_fields_k_{k}.png'))
    plt.close()  # Close the figure to save memory

print('\nAll ranks processed and histograms saved.')

# ==========================
# Fit Only for Final Rank (k=0)
# ==========================

k = k_final
print(f'\nPerforming curve fitting for final rank k={k}...')

# Retrieve local fields for k=0
lfs_final = results[k]['local_fields']

# Prepare data for fitting
hist, bin_edges = np.histogram(lfs_final, bins=bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Remove zero or negative bin centers (if necessary)
positive_indices = bin_centers > 0
xdata = bin_centers[positive_indices]
ydata = hist[positive_indices]

# Initial parameter guesses: a=1, b=1e-3
initial_guess = [1.0, 1e-3]

# Perform curve fitting
try:
    params, covariance = curve_fit(func1, xdata, ydata, p0=initial_guess, bounds=(0, np.inf))
    a_fit, b_fit = params
    print(f'Fitted parameters: a = {a_fit:.4f}, b = {b_fit:.4e}')
except RuntimeError as e:
    print(f'Curve fitting failed: {e}')
    params = [np.nan, np.nan]

# Plot histogram with fitted function
plt.figure(figsize=(10, 6))
sns.histplot(lfs_final, bins=bins, kde=False, stat='density', label='Data', color='gray', alpha=0.5)

if not np.isnan(params).any():
    # Generate x values for the fitted function
    x_fit = np.linspace(min(xdata), max(xdata), 500)
    y_fit = func1(x_fit, *params)

    # Plot the fitted function
    plt.plot(x_fit, y_fit, label=rf'Fit: $a x e^{{- a x^2 / 2}} + b$' + f'\n$a={a_fit:.2f}, b={b_fit:.2e}$',
             color='red')
else:
    print('Skipping fitted function plot due to fitting failure.')

plt.xlabel('Local Field Value')
plt.ylabel('Density')
plt.title(f'Histogram of Local Fields with Fitted Function for k={k}; N={N}, rho={rho}, beta={beta}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'local_fields_fit_final_rank.png'))
plt.show()

print('\nCurve fitting completed and plot saved.')
