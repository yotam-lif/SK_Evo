import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import numpy as np
import os

# Set some matplotlib parameters for a publication-like style
mpl.rcParams['font.family'] = 'Helvetica Neue'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

fig = plt.figure(figsize=(7, 6))
gs = fig.add_gridspec(2, 2, wspace=0.4, hspace=0.4)

########################################
# Subfigure A
########################################
axA = fig.add_subplot(gs[0, 0])
axA.set_title('A', loc='left', fontweight='bold', fontsize=12)

# Display the formula inside the subplot area
formula = r'$F(\vec{\sigma}) = \sum\limits_{i=1}^{N} \sigma_i h_i + \frac{1}{2} \sum\limits_{i,j}^{N} \sigma_i J_{ij} \sigma_j$'
axA.text(0.5, 0.95, formula, ha='center', va='top', transform=axA.transAxes, fontsize=12)

# Add the text \alpha_i \in {\pm 1 }
axA.text(0.5, 0.65, r'$\sigma_i \in \{\pm 1\}$', ha='center', va='top', transform=axA.transAxes, fontsize=12)

# Draw a series of spins as squares with arrows
L = 12
alpha = np.random.choice([-1, 1], size=L)
cmap = plt.get_cmap('CMRmap')
color_plus = cmap(0.4)  # Color for +1
color_minus = cmap(0.6)  # Color for -1
colors = [color_plus if a == 1 else color_minus for a in alpha]

y_base = 0.6
for i, a in enumerate(alpha):
    x_pos = i * 0.05
    # Rectangle for spin
    axA.add_patch(plt.Rectangle((x_pos, y_base), 0.05, 0.10, facecolor=colors[i], edgecolor='black', linewidth=1))
    # Arrow (up or down) with custom style
    if a == 1:
        axA.annotate('', xy=(x_pos + 0.025, y_base + 0.085), xytext=(x_pos + 0.025, y_base + 0.015),
                     arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', linewidth=1.0))
    else:
        axA.annotate('', xy=(x_pos + 0.025, y_base + 0.015), xytext=(x_pos + 0.025, y_base + 0.085),
                     arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', linewidth=1.0))

axA.set_xlim(-0.05, L * 0.05 + 0.002)
axA.set_ylim(0.5, 1.0)
axA.axis('off')

########################################
# Subfigure B
########################################
axB = fig.add_subplot(gs[0, 1])
axB.set_title('B', loc='left', fontweight='bold', fontsize=12)

# Create a fully connected graph
N = 175
rho = 0.05
d = int(N * rho)
G = nx.generators.random_graphs.random_regular_graph(d, N, seed=42)
pos = nx.spring_layout(G, seed=42)
# Node colors could represent h_i (random for demo)
h_values = np.random.choice([0, 1], size=N)

# Draw nodes with black edges and white or black inside
G_plus = [n for n in G.nodes() if h_values[n] == 1]
G_minus = [n for n in G.nodes() if h_values[n] == 0]
nx.draw_networkx_nodes(G_plus, pos, node_size=5, node_color='white', edgecolors='black', linewidths=0.25, ax=axB, label=r'$\sigma_i = +1$')
nx.draw_networkx_nodes(G_minus, pos, node_size=5, node_color='black', edgecolors='black', linewidths=0.25, ax=axB, label=r'$\sigma_i = -1$')

# Draw edges with varying opacity based on their strength and color by CMRmap
edges = G.edges()
edge_strengths = np.random.normal(loc=0.5, scale=0.1, size=len(edges))  # Gaussian strengths for demo
cmap = plt.get_cmap('CMRmap')
zipped = zip(edges, edge_strengths)
(u, v), strength = next(zipped)
nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=0.5, alpha=strength, edge_color=cmap(strength), ax=axB, label=r'$J_{ij}$')
for (u, v), strength in zipped:
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=0.5, alpha=strength, edge_color=cmap(strength), ax=axB)

# Add legend to the plot
axB.legend(loc='upper right', fontsize=8, markerscale=2, frameon=True)

axB.axis('off')

########################################
# Subfigure C
########################################
axC = fig.add_subplot(gs[1, 0])
axC.set_title('C', loc='left', fontweight='bold', fontsize=12)

# Conceptual free energy landscapes
alpha_range = np.linspace(-3, 3, 300)

# For beta=0, a single-peak landscape
F_beta0 = -np.exp(-0.5*alpha_range**2)

# For beta=1, a rugged landscape
np.random.seed(0)
centers = np.array([-2, -1, 0, 1.5, 2.5])
heights = np.array([1, 0.8, 1.2, 0.9, 1.1])
F_beta1 = np.zeros_like(alpha_range)
for c, h in zip(centers, heights):
    F_beta1 -= h*np.exp(-0.5*(alpha_range - c)**2 * 2)

# Plot landscapes
axC.plot(alpha_range[alpha_range<0], F_beta0[alpha_range<0], color='#00AAAA', linewidth=2, label=r'$\beta=0$')
axC.plot(alpha_range[alpha_range>0], F_beta1[alpha_range>0], color='#8844CC', linewidth=2, label=r'$\beta=1$')

# Add arrow indicating increasing beta
axC.annotate('increasing $\\beta$', xy=(0, -1.0), xytext=(-0.5, -3.0),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1),
             fontsize=10, ha='center')

axC.set_xlabel(r'$\alpha$')
axC.set_ylabel('F')
axC.legend(frameon=False, loc='upper left')
axC.spines.right.set_visible(False)
axC.spines.top.set_visible(False)

########################################
# Subfigure D (Placeholder)
########################################
axD = fig.add_subplot(gs[1, 1])
axD.set_title('D', loc='left', fontweight='bold', fontsize=12)

# Placeholder content, e.g., a simple text
axD.text(0.5, 0.5, 'Placeholder for D', ha='center', va='center', fontsize=12, transform=axD.transAxes)
axD.axis('off')

plt.tight_layout()

output_dir = '../Plots/paper_figs'
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, "model_exp.png"), dpi=800)