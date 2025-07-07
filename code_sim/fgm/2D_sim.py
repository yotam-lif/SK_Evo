import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cmn.cmn_fgm import Fisher  # import directly from uploaded file

plt.rcParams["font.family"] = "sans-serif"

# ─────────────────── Parameters ───────────────────
n          = 2
delta      = 0.1
m          = 1000
isotropic  = True
reps       = 5
max_steps  = 100
init_scale = 1.0

# Heatmap grid setup
domain    = 2.0
grid_size = 300
x = np.linspace(-domain, domain, grid_size)
y = np.linspace(-domain, domain, grid_size)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2))

# Plot heatmap
sns.set_style("white")
plt.figure(figsize=(8, 8))
plt.imshow(Z, origin='lower', extent=[-domain, domain, -domain, domain],
           cmap='plasma', aspect='equal', zorder=1)
cbar = plt.colorbar()
cbar.set_label('Fitness', rotation=270, labelpad=15)

# Adaptive walks with smaller circles
colors = sns.color_palette("hsv", reps)
for i in range(reps):
    model = Fisher(n=n, delta=delta, isotropic=isotropic, m=m, random_state=i)
    z0 = np.random.normal(scale=init_scale, size=n)
    flips, traj = model.relax(z0, max_steps=max_steps)
    traj = np.array(traj)

    plt.scatter(traj[:, 0], traj[:, 1],
                marker='o', facecolors='none', edgecolors=colors[i],
                s=20, linewidths=1, label=f'Rep {i+1}', zorder=2)

plt.legend(loc='upper right', frameon=False, labelcolor='white')
plt.tight_layout()
plt.show()