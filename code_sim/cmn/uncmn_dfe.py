import code_sim.cmn.cmn as cmn, code_sim.cmn.cmn_sk as cmn_sk
import numpy as np

def gen_final_dfe(N, beta, rho, num_repeats):
    """
    Generate final DFE histograms.

    Parameters
    ----------
    N : int
        N value.
    beta : float
        Epistasis strength.
    rho : float
        Fraction of non-zero coupling elements.
    num_repeats : int
        Number of repeats for each N value.

    Returns
    -------
    final_dfe : np.ndarray
        Final DFE.
    """
    final_dfe = np.array([])
    for _ in range(num_repeats):
        alpha_initial = cmn.init_sigma(N)
        h = cmn_sk.init_h(N, beta=beta)
        J = cmn_sk.init_J(N, beta=beta, rho=rho)
        flip_seq = cmn_sk.relax_sk(alpha_initial.copy(), h, J, sswm=True)
        final_alpha = cmn.compute_sigma_from_hist(alpha_initial, flip_seq)
        dfe = cmn_sk.compute_dfe(final_alpha, h, J)
        final_dfe = np.concatenate((final_dfe, dfe))
    return final_dfe.flatten()

def gen_bdfes(N, beta, rho, flip_list, num_repeats):
    """
    Generate BDFE histograms.
    Parameters
    ----------
    N: int
    beta: float
    rho: float
    flip_list: list(int)
    num_repeats: int

    Returns
    -------
    bdfes: list
    """
    bdfes = [[] for _ in range(flip_list)]

    for repeat in range(num_repeats):
        print(f"\nStarting repeat {repeat + 1} of {num_repeats}...")
        # Initialize spin configuration
        alpha_initial = cmn.init_sigma(N)

        # Initialize external fields
        h = cmn_sk.init_h(N, beta=beta)

        # Initialize coupling matrix with sparsity
        J = cmn_sk.init_J(N, beta=beta, rho=rho)

        # Relax the system using sswm_flip (sswm=True)
        flip_seq = cmn_sk.relax_sk(alpha_initial.copy(), h, J, sswm=True)
        alphas, _ = cmn.curate_sigma_list(alpha_initial, flip_seq, flip_list)
        for i, alpha in enumerate(alphas):
            # Calculate the BDFE for the current rank
            BDFE, _ = cmn_sk.compute_bdfe(alpha, h, J)
            bdfes[i].extend(BDFE)

    return bdfes





