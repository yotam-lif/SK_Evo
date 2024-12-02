import matplotlib.pyplot as plt
import numpy as np
import misc.cmn as cmn
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
    alpha_anc = cmn.compute_alpha_from_hist(alpha_init, flip_seq, anc_flip)
    alpha_evo = cmn.compute_alpha_from_hist(alpha_init, flip_seq, evo_flip)

    # Compute DFEs and BDFEs at the ancestor and evolved flips
    bdfe_anc, bdfe_anc_ind = cmn.calc_BDFE(alpha_anc, h, J)
    dfe_evo = cmn.calc_DFE(alpha_evo, h, J)
    prop_bdfe = dfe_evo[bdfe_anc_ind]

    return bdfe_anc, prop_bdfe

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
    alpha_anc = cmn.compute_alpha_from_hist(alpha_init, flip_seq, anc_flip)
    alpha_evo = cmn.compute_alpha_from_hist(alpha_init, flip_seq, evo_flip)

    # Compute DFEs and BDFEs at the ancestor and evolved flips
    bdfe_evo, bdfe_evo_ind = cmn.calc_BDFE(alpha_evo, h, J)
    dfe_anc = cmn.calc_DFE(alpha_anc, h, J)
    prop_bdfe = dfe_anc[bdfe_evo_ind]

    return bdfe_evo, prop_bdfe


def gen_crossings(ax, alpha_init, h, J, flip_seq, anc_flip, evo_flip):
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
    bdfe1, prop_bdfe1 = propagate_forward(alpha_init, h, J, flip_seq, anc_flip, evo_flip)
    bdfe2, prop_bdfe2 = propagate_backward(alpha_init, h, J, flip_seq, anc_flip, evo_flip)

    # Prepare data for plotting
    flips_labels = [f"flip {anc_flip}", f"flip {evo_flip}"]

    # Plot ancestral DFEs
    color_anc = 'coral'
    color_evo = 'royalblue'
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
    ax.set_xlabel("Flip")
    ax.set_ylabel("$\\Delta$")
    ax.legend(fontsize=12, frameon=True)
    ax.figure.tight_layout()
