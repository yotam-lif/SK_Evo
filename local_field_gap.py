import os
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 4000  # Number of spins
k = 100  # Desired number of spins with positive energy effect
bins = 30  # Number of bins for histograms
output_dir = 'figures'  # Directory to save figures

# Create directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Initialize J matrix with entries drawn from a normal distribution
J = np.random.normal(0, 1, (N, N))

# Initialize spins randomly as +1 or -1
S = np.random.choice([-1, 1], size=N)


def local_fields(S, J):
    return np.dot(J, S)

# Function to flip a spin based on a probability vector
def flip(S, J):
    # Normalize the energy change distribution to get probabilities
    DFE = -2 * S * local_fields(S, J)
    P = np.where(DFE > 0, DFE, 0)
    P /= np.sum(P)
    flip_index = np.random.choice(np.arange(len(S)), p=P)
    S[flip_index] *= -1  # Flip the spin
    return S


# Function to flip spins until only k spins remain with a positive energy effect
def flip_until_rank_k(S, J, k):
    remaining_spins = N  # Start with all spins considered
    BDFE_indices = []  # Indices of spins with positive energy effect
    while remaining_spins > k:
        lf = local_fields(S, J)
        DFE = -2 * S * lf
        BDFE_indices = np.where(DFE > 0)[0]

        # If more than k spins have a positive energy effect, flip one based on probability
        if len(BDFE_indices) > k:
            S = flip(S, J)
            remaining_spins -= 1
        else:
            break

    return S, BDFE_indices


# Step 1: Flip spins until only k remain with a positive energy effect
S, BDFE_indices = flip_until_rank_k(S, J, k)

# Step 2: Perform k iterations and plot the distribution of local fields
for iteration in range(k):
    lfs = local_fields(S, J)
    DFE = -2 * S * lfs

    # Flip one spin based on probability
    S = flip(S, J)

    if iteration % 20 == 0 or iteration == k - 1:
        plt.figure(figsize=(12, 5))

        # Plot distribution of local fields h_i
        plt.subplot(1, 2, 1)
        plt.hist(lfs, bins=bins, density=True, alpha=0.75)
        plt.xlabel(r'$h_i$')
        plt.ylabel('Probability Density')
        plt.title(f'Distribution of Local Fields $h_i$ at Iteration {iteration}')

        # Plot distribution of energy-change -2*S_i*h_i
        plt.subplot(1, 2, 2)
        plt.hist(DFE, bins=bins, density=True, alpha=0.75)
        plt.xlabel(r'Energy Change $-2S_ih_i$')
        plt.ylabel('Probability Density')
        plt.title(f'Energy Change Distribution at Iteration {iteration}')

        # Save the figure in the specified directory
        plt.savefig(os.path.join(output_dir, f'local_fields_energy_change_iteration_{iteration}.png'))
        plt.close()

# Output the indices of the spins with positive energy effect
print("Indices of spins with positive energy effect after reaching rank k:", BDFE_indices)