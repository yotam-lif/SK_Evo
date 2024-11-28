# Import the Funcs module
from misc import cmn
import numpy as np
import matplotlib.pyplot as plt
import os


def gen_final_dfes(N_arr, beta, rho, num_repeats):
    """
    Generate final DFE histograms.

    Parameters
    ----------
    N_arr : list
        List of N values.
    beta : float
        Epistasis strength.
    rho : float
        Fraction of non-zero coupling elements.
    num_repeats : int
        Number of repeats for each N value.

    Returns
    -------
    final_dfes : list of np.ndarray
        List of final DFEs for each N value.
    """
    final_dfes = []
    for N in N_arr:
        print(f"\nStarting N={N}...")
        final_dfes_N = np.empty(0, dtype=float)
        for _ in range(num_repeats):
            alpha_initial = cmn.init_alpha(N)
            h = cmn.init_h(N, beta=beta)
            J = cmn.init_J(N, beta=beta, rho=rho)
            flip_seq = cmn.relax_sk(alpha_initial.copy(), h, J, sswm=True)
            final_alpha = cmn.compute_alpha_from_hist(alpha_initial, flip_seq)
            dfe = cmn.calc_DFE(final_alpha, h, J)
            final_dfes_N = np.concatenate((final_dfes_N, dfe))
        final_dfes.append(final_dfes_N)
    return final_dfes

def gen_bdfes(N, beta, rho, num_points, num_repeats):
    """
    Generate BDFE histograms.
    Parameters
    ----------
    N: int
    beta: float
    rho: float
    num_points: int
    num_repeats: int

    Returns
    -------
    bdfes: list
    """
    bdfes = [[] for _ in range(num_points)]

    for repeat in range(num_repeats):
        print(f"\nStarting repeat {repeat + 1} of {num_repeats}...")
        # Initialize spin configuration
        alpha_initial = cmn.init_alpha(N)

        # Initialize external fields
        h = cmn.init_h(N, beta=beta)

        # Initialize coupling matrix with sparsity
        J = cmn.init_J(N, beta=beta, rho=rho)

        # Relax the system using sswm_flip (sswm=True)
        flip_seq = cmn.relax_sk(alpha_initial.copy(), h, J, sswm=True)
        alphas, _ = cmn.curate_alpha_list(alpha_initial, flip_seq, num_points)
        for i, alpha in enumerate(alphas):
            # Calculate the BDFE for the current rank
            BDFE, _ = cmn.calc_BDFE(alpha, h, J)
            bdfes[i].extend(BDFE)

    return bdfes



