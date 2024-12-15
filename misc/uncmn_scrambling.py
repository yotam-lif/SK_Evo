import matplotlib.pyplot as plt
import numpy as np
from misc import cmn, cmn_sk
# import os

def propagate_forward(alpha_init, h, J, flip_seq, anc_flip, evo_flip):
    """
    Propagate the DFE forward from the ancestor flip to the evolved flip.

    Parameters:
        alpha_init (np.ndarray): Initial alpha vector.
        h (np.ndarray): External field vector.
        J (np.ndarray): Coupling matrix.
        flip_seq (np.ndarray): Sequence of flips.
        anc_flip (int): Index of the ancestor flip.
        evo_flip (int): Index of the evolved flip.

    Returns:
        bdfe_anc (np.ndarray): BDFE at the ancestor flip.
        prop_bdfe (np.ndarray): Propagated BDFE at the evolved flip.
    """
    # Compute alpha vectors at the ancestor and evolved flips
    alpha_anc = cmn.compute_sigma_from_hist(alpha_init, flip_seq, anc_flip)
    alpha_evo = cmn.compute_sigma_from_hist(alpha_init, flip_seq, evo_flip)

    # Compute DFEs and BDFEs at the ancestor and evolved flips
    bdfe_anc, bdfe_anc_ind = cmn_sk.calc_BDFE(alpha_anc, h, J)
    dfe_evo = cmn_sk.calc_DFE(alpha_evo, h, J)
    prop_bdfe = dfe_evo[bdfe_anc_ind]

    return bdfe_anc, prop_bdfe, dfe_evo

def propagate_backward(alpha_init, h, J, flip_seq, anc_flip, evo_flip):
    """
    Propagate the DFE backward from the evolved flip to the ancestor flip.

    Parameters:
        alpha_init (np.ndarray): Initial alpha vector.
        h (np.ndarray): External field vector.
        J (np.ndarray): Coupling matrix.
        flip_seq (np.ndarray): Sequence of flips.
        anc_flip (int): Index of the ancestor flip.
        evo_flip (int): Index of the evolved flip.

    Returns:
        bdfe_evo (np.ndarray): BDFE at the evolved flip.
        prop_bdfe (np.ndarray): Propagated BDFE at the ancestor flip.
    """
    # Compute alpha vectors at the ancestor and evolved flips
    alpha_anc = cmn.compute_sigma_from_hist(alpha_init, flip_seq, anc_flip)
    alpha_evo = cmn.compute_sigma_from_hist(alpha_init, flip_seq, evo_flip)

    # Compute DFEs and BDFEs at the ancestor and evolved flips
    bdfe_evo, bdfe_evo_ind = cmn_sk.calc_BDFE(alpha_evo, h, J)
    dfe_anc = cmn_sk.calc_DFE(alpha_anc, h, J)
    prop_bdfe = dfe_anc[bdfe_evo_ind]

    return bdfe_evo, prop_bdfe, dfe_anc


def gen_crossings(ax, alpha_init, h, J, flip_seq, anc_flip, evo_flip, color1, color2):
    """
    Generate and plot crossings between two specific flips.

    Parameters:
        ax (plt.Axes): The matplotlib axis to plot on.
        alpha_init (np.ndarray): Initial alpha vector.
        h (np.ndarray): External field vector.
        J (np.ndarray): Coupling matrix.
        flip_seq (np.ndarray): Sequence of flips.
        anc_flip (int): Index of the ancestor flip.
        evo_flip (int): Index of the evolved flip.
        color1: Color for the ancestral DFEs.
        color2: Color for the evolved DFEs.

    Returns:
        None
    """
    # Extract proposed DFEs based on indices
    bdfe1, prop_bdfe1, _ = propagate_forward(alpha_init, h, J, flip_seq, anc_flip, evo_flip)
    bdfe2, prop_bdfe2, _ = propagate_backward(alpha_init, h, J, flip_seq, anc_flip, evo_flip)
    num_flips = len(flip_seq)
    flip_anc_percent = int(anc_flip * 100 / num_flips)
    flip_evo_percent = int(evo_flip * 100 / num_flips)

    # Prepare data for plotting
    flips_labels = [f"{flip_anc_percent} \\%", f"{flip_evo_percent} \\%"]

    # Plot ancestral DFEs
    color_anc = color1
    color_evo = color2
    for j in range(len(bdfe1)):
        ax.plot(flips_labels, [bdfe1[j], prop_bdfe1[j]], color=color_anc, alpha=0.075)
    ax.scatter([flips_labels[0]] * len(bdfe1), bdfe1, color=color_anc, edgecolor=color_anc, s=20, facecolors='none', label='Forwards')
    ax.scatter([flips_labels[1]] * len(prop_bdfe1), prop_bdfe1, color=color_anc, edgecolor=color_anc, s=20, facecolors='none')

    # Plot evolved DFEs
    for j in range(len(bdfe2)):
        ax.plot(flips_labels, [prop_bdfe2[j], bdfe2[j]], color=color_evo, alpha=0.075)
    ax.scatter([flips_labels[0]] * len(prop_bdfe2), prop_bdfe2, color=color_evo, edgecolor=color_evo, s=20, facecolors='none', label='Backwards')
    ax.scatter([flips_labels[1]] * len(bdfe2), bdfe2, color=color_evo, edgecolor=color_evo, s=20, facecolors='none')

    # Customize plot
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("\\% of walk completed", fontsize=14)
    ax.set_ylabel("$\\Delta$", fontsize=14)
    ax.legend(fontsize=12, frameon=True)
    ax.set_xticks([])
    ax.figure.tight_layout()
