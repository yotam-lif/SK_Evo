import numpy as np
from matplotlib import pyplot as plt
import Funcs as fs

# Parameters
exps = 1000
N = 500
frustration_count = []

for i in range(exps):
    print(i)
    J = fs.init_J(N)
    alpha = np.random.choice([-1, 1], size=N)
    pf = fs.sswm_prob
    alpha_max = fs.relax_SK(alpha, J, pf)
    hamiltonian_terms = np.outer(alpha_max, alpha_max) * J
    frustration_count.append(np.sum(hamiltonian_terms < 0))

frustration_count = np.array(frustration_count) / N ** 2

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(frustration_count, bins=50, color='skyblue', edgecolor='black', density=True)
plt.title('Distribution of Frustration')
plt.xlabel('Frustration Fraction')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
