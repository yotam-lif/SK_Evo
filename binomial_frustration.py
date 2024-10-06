import numpy as np
from matplotlib import pyplot as plt
import Funcs as fs

# Parameters
N = 1000
J = fs.init_J(N)
alpha = np.random.choice([-1, 1], size=N)
lf_frustration_count = []
pf = fs.sswm_prob
alpha = fs.relax_SK(alpha, J, pf)

hamiltonian_terms = np.outer(alpha, alpha) * J
for i in range(N):
    lf_frustration_count.append(np.sum(hamiltonian_terms[i] < 0))

# Convert h_dist to a NumPy array for efficient processing
lf_frustration_count = np.array(lf_frustration_count)

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(lf_frustration_count, bins=75, color='skyblue', edgecolor='black', density=True)
plt.title('Distribution of h_dist')
plt.xlabel('Sum Difference')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
