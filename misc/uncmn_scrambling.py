import matplotlib.pyplot as plt
from misc import cmn
import numpy as np
# import os

def plot_crossings(flips_labels, anc_bdfe, prop_anc_bdfe, evol_bdfe, prop_evol_bdfe, N, beta, rho):
    """
    Plot crossings between two consecutive flips.

    Parameters:
        flips_labels (list): Labels for the two flips.
        anc_bdfe (np.ndarray): Ancestral BDFE values.
        prop_anc_bdfe (np.ndarray): Proposed Ancestral BDFE values.
        evol_bdfe (np.ndarray): Evolved BDFE values.
        prop_evol_bdfe (np.ndarray): Proposed Evolved BDFE values.
        N (int): Number of spins (for plot title).
        beta (float): Epistasis strength.
        rho (float): Fraction of non-zero coupling elements.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot ancestral DFEs
    for j in range(len(anc_bdfe)):
        ax.plot(flips_labels, [anc_bdfe[j], prop_anc_bdfe[j]], color='coral', alpha=0.2)
    ax.scatter([flips_labels[0]] * len(anc_bdfe), anc_bdfe, color='coral', edgecolor='coral', s=20, facecolors='none', label='Forwards')
    ax.scatter([flips_labels[1]] * len(prop_anc_bdfe), prop_anc_bdfe, color='coral', edgecolor='coral', s=20, facecolors='none')

    # Plot evolved DFEs
    for j in range(len(evol_bdfe)):
        ax.plot(flips_labels, [prop_evol_bdfe[j], evol_bdfe[j]], color='royalblue', alpha=0.2)
    ax.scatter([flips_labels[0]] * len(prop_evol_bdfe), prop_evol_bdfe, color='royalblue', edgecolor='royalblue', s=20, facecolors='none', label='Backwards')
    ax.scatter([flips_labels[1]] * len(evol_bdfe), evol_bdfe, color='royalblue', edgecolor='royalblue', s=20, facecolors='none')

    # Customize plot
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Flip")
    ax.set_ylabel("Fitness (s)")
    ax.set_title(f"Crossings between {flips_labels[0]} - {flips_labels[1]}; N = {N}; β = {beta}; ρ = {rho}")
    ax.legend()
    fig.tight_layout()

    return fig

def gen_crossings_1(N, beta, rho, num_stops):
    alpha_initial = cmn.init_alpha(N)
    h = cmn.init_h(N, beta=beta)
    J = cmn.init_J(N, beta=beta, rho=rho)

    # Relax the system using sswm_flip (sswm=True)
    flip_seq = cmn.relax_sk(alpha_initial.copy(), h, J, sswm=True)
    flip_num = len(flip_seq)
    flips = np.linspace(0, flip_num, num=num_stops, dtype=int)

    # Ensure that flip indices are unique and sorted
    flips = sorted(set(flips))
    num_stops = len(flips)

    # Compute saved alphas
    saved_alphas = [cmn.compute_alpha_from_hist(alpha_initial, flip_seq, flip) for flip in flips]

    dfes = []
    bdfes = []
    for alpha in saved_alphas:
        dfe = cmn.calc_DFE(alpha, h, J)
        dfes.append(dfe)
        bdfe, bdfe_ind = cmn.calc_BDFE(alpha, h, J)
        bdfes.append((bdfe, bdfe_ind))

    # Plot each pair of consecutive DFEs
    for i in range(num_stops - 1):
        anc = dfes[i]
        evol = dfes[i + 1]
        anc_bdfe, anc_bdfe_ind = bdfes[i]
        evol_bdfe, evol_bdfe_ind = bdfes[i + 1]

        # Extract proposed DFEs based on indices
        prop_anc_bdfe = evol[anc_bdfe_ind]
        prop_evol_bdfe = anc[evol_bdfe_ind]

        # Prepare data for plotting
        flips_labels = [f"flip {flips[i]}", f"flip {flips[i + 1]}"]

        # Plot crossings for the current rank pair
        plot_crossings(
            flips_labels,
            anc_bdfe,
            prop_anc_bdfe,
            evol_bdfe,
            prop_evol_bdfe,
            N,
            beta,
            rho
        )


def gen_crossings(ax, dfe1, dfe2, bdfe1, bdfe2, bdfe1_ind, bdfe2_ind, flip1, flip2):
    """
    Generate and plot crossings between two specific flips.

    Parameters:
        ax (plt.Axes): The matplotlib axis to plot on.
        dfe1 (np.ndarray): DFE at the first flip.
        dfe2 (np.ndarray): DFE at the second flip.
        bdfe1 (np.ndarray): BDFE at the first flip.
        bdfe2 (np.ndarray): BDFE at the second flip.
        bdfe1_ind (np.ndarray): Indices of BDFE at the first flip.
        bdfe2_ind (np.ndarray): Indices of BDFE at the second flip.
        flip1 (int): Index of the first flip.
        flip2 (int): Index of the second flip.
    """
    # Extract proposed DFEs based on indices
    prop_bdfe1 = dfe2[bdfe1_ind]
    prop_bdfe2 = dfe1[bdfe2_ind]

    # Prepare data for plotting
    flips_labels = [f"flip {flip1}", f"flip {flip2}"]

    # Plot ancestral DFEs
    color_anc = 'coral'
    color_evo = 'royalblue'
    for j in range(len(bdfe1)):
        ax.plot(flips_labels, [bdfe1[j], prop_bdfe1[j]], color=color_anc, alpha=0.2)
    ax.scatter([flips_labels[0]] * len(bdfe1), bdfe1, color=color_anc, edgecolor=color_anc, s=20, facecolors='none', label='Forwards')
    ax.scatter([flips_labels[1]] * len(prop_bdfe1), prop_bdfe1, color=color_anc, edgecolor=color_anc, s=20, facecolors='none')

    # Plot evolved DFEs
    for j in range(len(bdfe2)):
        ax.plot(flips_labels, [prop_bdfe2[j], bdfe2[j]], color=color_evo, alpha=0.2)
    ax.scatter([flips_labels[0]] * len(prop_bdfe2), prop_bdfe2, color=color_evo, edgecolor=color_evo, s=20, facecolors='none', label='Backwards')
    ax.scatter([flips_labels[1]] * len(bdfe2), bdfe2, color=color_evo, edgecolor=color_evo, s=20, facecolors='none')

    # Customize plot
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Flip")
    ax.set_ylabel("Fitness (s)")
    ax.legend()
    ax.set_title(f"Crossings between {flips_labels[0]} - {flips_labels[1]}")
    ax.figure.tight_layout()
