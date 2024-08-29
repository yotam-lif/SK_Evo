import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1000  # Number of spins
k = 100  # Number of random flips after reaching extremum

# Initialize J matrix with entries drawn from a normal distribution
J = np.random.normal(0, 1, (N, N))

# Initialize spins randomly as +1 or -1
S = np.random.choice([-1, 1], size=N)


# Function to calculate the local fields h_i
def calculate_local_fields(J, S):
    return np.dot(J, S)


# Function to find local extremum more efficiently
def reach_local_extremum(S, J):
    extremum_reached = False
    flipped_spins = []

    while not extremum_reached:
        extremum_reached = True
        local_fields = calculate_local_fields(J, S)

        for i in range(N):
            if S[i] * local_fields[i] < 0:  # If S_i has opposite sign to h_i
                S[i] *= -1  # Flip the spin
                flipped_spins.append(i)
                extremum_reached = False
                break

    return S, flipped_spins


# Step 1: Reach a local extremum
S, flipped_spins = reach_local_extremum(S, J)

# Step 2: Perform k random flips to escape the extremum
for _ in range(k):
    i = np.random.randint(0, N)
    S[i] *= -1  # Randomly flip a spin

# Step 3: Iteratively flip spins that were flipped to escape the extremum
local_fields_list = []
energy_changes = []

for i in flipped_spins:
    local_fields = calculate_local_fields(J, S)
    S_new = S.copy()
    S_new[i] *= -1  # Flip the spin
    energy_change = abs(np.dot(local_fields, S_new) - np.dot(local_fields, S))
    energy_changes.append(energy_change)

energy_changes = np.array(energy_changes)
probabilities = energy_changes / np.sum(energy_changes)  # Normalize energy changes

# Perform k iterations and save histogram every 10 iterations
for iteration in range(k):
    local_fields = calculate_local_fields(J, S)
    local_fields_list.append(local_fields)

    for i, prob in zip(flipped_spins, probabilities):
        if np.random.rand() < prob:  # Flip the spin with the corresponding probability
            S[i] *= -1

    if iteration % 20 == 0:
        plt.hist(local_fields, bins=30, density=True, alpha=0.75)
        plt.xlabel(r'$h_i$')
        plt.ylabel('Probability Density')
        plt.title(f'Distribution of Local Fields $h_i$ at Iteration {iteration}')
        plt.savefig(f'local_fields_iteration_{iteration}.png')
        plt.close()
