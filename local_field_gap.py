import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1000  # Number of spins
k = 200  # Desired number of spins with positive energy effect
bins = 50  # Number of bins for histograms

# Initialize J matrix with entries drawn from a normal distribution
J = np.random.normal(0, 1, (N, N))

# Initialize spins randomly as +1 or -1
S = np.random.choice([-1, 1], size=N)


# Function to calculate the local fields h_i
def calculate_local_fields(J, S):
    return np.dot(J, S)


# Function to flip spins until only k spins remain with a positive energy effect
def flip_until_rank_k(S, J, k):
    remaining_spins = N  # Start with all spins considered
    while remaining_spins > k:
        local_fields = calculate_local_fields(J, S)
        energy_change_distribution = -2 * S * local_fields
        positive_effect_indices = np.where(energy_change_distribution > 0)[0]

        # If more than k spins have a positive energy effect, flip one randomly
        if len(positive_effect_indices) > k:
            flip_index = np.random.choice(positive_effect_indices)
            S[flip_index] *= -1  # Flip the spin
            remaining_spins -= 1
        else:
            break

    return S, positive_effect_indices


# Step 1: Flip spins until only k remain with a positive energy effect
S, positive_effect_indices = flip_until_rank_k(S, J, k)

# Step 2: Perform k iterations and plot the distribution of local fields
for iteration in range(k):
    local_fields = calculate_local_fields(J, S)
    energy_change_distribution = -2 * S * local_fields

    if iteration % 20 == 0 or iteration == k - 1:
        plt.figure(figsize=(12, 5))

        # Plot distribution of local fields h_i
        plt.subplot(1, 2, 1)
        plt.hist(local_fields, bins=bins, density=True, alpha=0.75)
        plt.xlabel(r'$h_i$')
        plt.ylabel('Probability Density')
        plt.title(f'Distribution of Local Fields $h_i$ at Iteration {iteration}')

        # Plot distribution of energy-change -2*S_i*h_i
        plt.subplot(1, 2, 2)
        plt.hist(energy_change_distribution, bins=bins, density=True, alpha=0.75)
        plt.xlabel(r'Energy Change $-2S_ih_i$')
        plt.ylabel('Probability Density')
        plt.title(f'Energy Change Distribution at Iteration {iteration}')

        plt.savefig(f'local_fields_energy_change_iteration_{iteration}.png')
        plt.close()

# Output the indices of the spins with positive energy effect
print("Indices of spins with positive energy effect after reaching rank k:", positive_effect_indices)
