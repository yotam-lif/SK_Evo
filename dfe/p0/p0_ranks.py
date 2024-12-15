import os
import numpy as np
import matplotlib.pyplot as plt
from misc.cmn_sk import init_alpha, init_h, init_J, relax_sk, calc_DFE, curate_alpha_list, calc_rank, compute_fit_slow, calc_F_off
import scienceplots

# Parameters
N = 3000  # Number of spins
beta = 1.0  # Epistasis strength
rho = 1.0  # Fraction of non-zero coupling elements
random_state = 42  # Seed for reproducibility
num_points = 100
plt.style.use('science')
plt.figure()

plt.gca().invert_xaxis()

# Initialize the model
alpha = init_alpha(N)
h = init_h(N, random_state=random_state, beta=beta)
J = init_J(N, random_state=random_state, beta=beta, rho=rho)

# Perform relaxation and save alphas at different time points
flip_seq = relax_sk(alpha.copy(), h, J, sswm=True)
alpha_list, time_points = curate_alpha_list(alpha, flip_seq, num_points)


# Create directory for saving histograms
output_dir = "../../Plots/p0_dynamics"
os.makedirs(output_dir, exist_ok=True)

# Initialize list to store the size of the bin with delta=0
bin_sizes = []

# Iterate over saved alphas and corresponding time points
for alpha_i in alpha_list:
    if alpha_i is not None:
        # Calculate BDFE
        DFE = calc_DFE(alpha_i, h, J)
        # Create histogram
        hist, bin_edges = np.histogram(DFE, bins=75, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find the bin containing delta=0
        zero_bin_index = np.digitize(0, bin_edges) - 1
        zero_bin_size = hist[zero_bin_index] if 0 <= zero_bin_index < len(hist) else 0
        bin_sizes.append(zero_bin_size)

r_list = []
for alpha in alpha_list:
    r_list.append(calc_rank(alpha, h, J)/N)

# Plot the size of the bin with delta=0 as a function of time
plt.plot(r_list, bin_sizes, 'o-', markersize=3)
plt.ylabel('$P(0, t)$')
plt.xlabel('$r(t)/N$')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'P0_rank.png'), dpi=300)
plt.close()