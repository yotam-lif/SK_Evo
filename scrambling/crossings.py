import matplotlib.pyplot as plt
from misc import cmn
import numpy as np

def gen_crossings(N, beta, rho, num_stops):
    """
    Generate and plot crossings between two consecutive flips.

    Parameters:
        N (int): Number of spins.
        beta (float): Epistasis strength.
        rho (float): Fraction of non-zero coupling elements.
        num_stops (int): Number of stops (flips) to consider.

    Returns:
        None
    """
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

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        color_anc = 'coral'
        color_evo = 'royalblue'

        # Plot ancestral DFEs
        for j in range(len(anc_bdfe)):
            ax.plot(flips_labels, [anc_bdfe[j], prop_anc_bdfe[j]], color=color_anc, alpha=0.2)
        ax.scatter([flips_labels[0]] * len(anc_bdfe), anc_bdfe, color=color_anc, edgecolor=color_anc, s=20, facecolors='none', label='Forwards')
        ax.scatter([flips_labels[1]] * len(prop_anc_bdfe), prop_anc_bdfe, color=color_anc, edgecolor=color_anc, s=20, facecolors='none')

        # Plot evolved DFEs
        for j in range(len(evol_bdfe)):
            ax.plot(flips_labels, [prop_evol_bdfe[j], evol_bdfe[j]], color=color_evo, alpha=0.2)
        ax.scatter([flips_labels[0]] * len(prop_evol_bdfe), prop_evol_bdfe, color=color_evo, edgecolor=color_evo, s=20, facecolors='none', label='Backwards')
        ax.scatter([flips_labels[1]] * len(evol_bdfe), evol_bdfe, color=color_evo, edgecolor=color_evo, s=20, facecolors='none')

        # Customize plot
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Flip")
        ax.set_ylabel("Fitness (s)")
        ax.set_title(f"Crossings between {flips_labels[0]} - {flips_labels[1]}; N = {N}; β = {beta}; ρ = {rho}")
        ax.legend()
        fig.tight_layout()

        return fig