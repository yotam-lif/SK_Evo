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

fig = plt.figure(figsize=(21, 6))
gs = fig.add_gridspec(1, 3, wspace=0.1)

########################################
# Subfigure A
########################################
axA = fig.add_subplot(gs[0, 0])
axA.set_title('A', loc='left', fontweight='heavy', fontsize=20)

# Display the formula inside the subplot area
formula = r'$F(\vec{\sigma}) = \sum\limits_{i=1}^{N} \sigma_i h_i + \frac{1}{2} \sum\limits_{i,j}^{N} \sigma_i J_{ij} \sigma_j$'
axA.text(0.5, 0.95, formula, ha='center', va='top', transform=axA.transAxes, fontsize=24)

# Add the text \alpha_i \in {\pm 1 }
axA.text(0.5, 0.7, r'$\sigma_i \in \{\pm 1\}$', ha='center', va='top', transform=axA.transAxes, fontsize=24)

# Draw a series of spins as squares with arrows
L = 11
alpha = np.random.choice([-1, 1], size=L)
cmap = plt.get_cmap('CMRmap')
color_plus = cmap(0.6)  # Color for +1
color_minus = cmap(0.4)  # Color for -1
colors = [color_plus if a == 1 else color_minus for a in alpha]

y_base = 0.65
rect_height = 0.06
width = 0.05

for i, a in enumerate(alpha):
    x_pos = i * width
    # Draw rectangle for spin
    axA.add_patch(plt.Rectangle((x_pos, y_base), width, rect_height,
                                facecolor=colors[i], edgecolor='black', linewidth=1.0))

    # Center coordinates inside the rectangle
    x_center = x_pos + width / 2.0
    # For an upward spin, arrow points upward inside the rectangle
    y_margin = 0.01
    if a == 1:
        axA.annotate('',
                     xy=(x_center, y_base + rect_height - y_margin),
                     xytext=(x_center, y_base + y_margin),
                     arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', linewidth=1.4))
    else:
        # For a downward spin, arrow points downward inside the rectangle
        axA.annotate('',
                     xy=(x_center, y_base + y_margin),  # arrow head near bottom
                     xytext=(x_center, y_base + rect_height - y_margin),  # arrow tail near top
                     arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', linewidth=1.4))

axA.set_xlim(-0.05, L * 0.05 + 0.005)
axA.set_ylim(0.5, 1.0)
axA.axis('off')

########################################
# Subfigure B
########################################
axB = fig.add_subplot(gs[0, 1])
axB.set_title('B', loc='left', fontweight='heavy', fontsize=20)

# Create a fully connected random regular graph (from your original code)
N = 100
rho = 0.05
d = int(N * rho)
G = nx.generators.random_graphs.random_regular_graph(d, N, seed=42)
pos = nx.spring_layout(G, seed=42)

# Node colors represent h_i (0 or 1)
h_values = np.random.choice([0, 1], size=N)
G_plus = [n for n in G.nodes() if h_values[n] == 1]
G_minus = [n for n in G.nodes() if h_values[n] == 0]

# Draw nodes
nx.draw_networkx_nodes(G, pos, nodelist=G_plus, node_size=20, node_color='white', edgecolors='black', linewidths=0.5, ax=axB, label=r'$\sigma_i = +1$')
nx.draw_networkx_nodes(G, pos, nodelist=G_minus, node_size=20, node_color='black', edgecolors='black', linewidths=0.5, ax=axB, label=r'$\sigma_i = -1$')

# Generate J_ij values for edges from a Gaussian distribution
edges = list(G.edges())
J_values = np.random.normal(loc=0.0, scale=1.0, size=len(edges))

# We'll pick two colors from the colormap for positive/negative J
cmap = plt.get_cmap('CMRmap')
pos_color = cmap(0.5)
neg_color = cmap(0.2)

# Determine max absolute value for scaling alpha
max_abs_J = np.max(np.abs(J_values))

# Separate edges into positive and negative lists for legend
pos_edges = [(e, j) for e, j in zip(edges, J_values) if j > 0]
neg_edges = [(e, j) for e, j in zip(edges, J_values) if j < 0]

# Draw one edge for each category for legend handles
pos_legend_edge = [(0,1)]  # dummy edge for legend
neg_legend_edge = [(1,2)]  # dummy edge for legend

# Plot positive edges
for (u, v), j in pos_edges:
    alpha_val = min(1.0, abs(j) / max_abs_J)
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=0.7, alpha=alpha_val, edge_color=pos_color, ax=axB)

# Plot negative edges
for (u, v), j in neg_edges:
    alpha_val = min(1.0, abs(j) / max_abs_J)
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=0.7, alpha=alpha_val, edge_color=neg_color, ax=axB)

# Create legend entries for positive and negative edges
pos_line = plt.Line2D([0], [0], color=pos_color, alpha=0.8, lw=2, label=r'$J_{ij} > 0$')
neg_line = plt.Line2D([0], [0], color=neg_color, alpha=0.8, lw=2, label=r'$J_{ij} < 0$')
plus_node = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='black', markersize=5, label=r'$\sigma_i = +1$')
minus_node = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markeredgecolor='black', markersize=5, label=r'$\sigma_i = -1$')

axB.legend(handles=[plus_node, minus_node, pos_line, neg_line], loc='lower right', fontsize=16, markerscale=2, frameon=True)

axB.axis('off')

########################################
# Subfigure C
########################################
axC = fig.add_subplot(gs[0, 2])
axC.set_title('C', loc='left', fontweight='heavy', fontsize=20)

beta0_xmin = -1.0
beta0_xmax = -0.025
beta1_xmin = 0.025
beta1_xmax = 1.0

xrange0 = np.linspace(beta0_xmin, beta0_xmax, 200)
xrange1 = np.linspace(beta1_xmin, beta1_xmax, 200)

# For beta=0: a simple Gaussian peak
sig_0 = 0.1
F_beta0 = 0.8 * (1/sig_0/np.sqrt(2*np.pi))*np.exp(-(xrange0 + 0.5)**2 / (2*sig_0**2))

# For beta=1: a rugged landscape as a superposition of multiple Gaussians
N = 6
np.random.seed(6)
centers = np.random.uniform(beta1_xmin + 0.2, beta1_xmax - 0.1, size=N)
heights = np.random.normal(2.0, 0.5, size=N)
widths = np.abs(np.random.normal(0.04, 0.001, size=N))  # ensure widths are positive

F_beta1 = np.zeros_like(xrange1)
for c, h, w in zip(centers, heights, widths):
    # Gaussian: h * exp(-((alpha-c)^2 / (2*w^2)))
    F_beta1 += h * np.exp(-0.5 * ((xrange1 - c)/w)**2)

# Plot both landscapes
axC.plot(xrange0, F_beta0, color=cmap(0.5), linewidth=2, label=r'$\beta=0$')
axC.plot(xrange1, F_beta1, color=cmap(0.2), linewidth=2, label=r'$\beta=1$')

# Add annotation for increasing beta
axC.annotate('increasing $\\beta$', xy=(0, -1.0), xytext=(-0.5, -3.0),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1),
             fontsize=10, ha='center')

axC.set_ylabel(r'$F(\vec{\sigma})$', fontsize=18)
axC.set_xlabel(r'$\vec{\sigma}$', fontsize=18)
axC.legend(frameon=True, loc='upper right', fontsize=18)
axC.spines.right.set_visible(False)
axC.spines.top.set_visible(False)
axC.set_xticklabels([])
axC.set_yticklabels([])
axC.tick_params(axis='both', which='both', length=0)

output_dir = '../Plots/paper_figs'
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, "model_exp.png"), dpi=800, bbox_inches='tight')