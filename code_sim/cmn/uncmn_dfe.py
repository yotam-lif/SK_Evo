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
    bdfe_anc, bdfe_anc_ind = cmn_sk.compute_bdfe(alpha_anc, h, J)
    dfe_evo = cmn_sk.compute_dfe(alpha_evo, h, J)
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
    bdfe_evo, bdfe_evo_ind = cmn_sk.compute_bdfe(alpha_evo, h, J)
    dfe_anc = cmn_sk.compute_dfe(alpha_anc, h, J)
    prop_bdfe = dfe_anc[bdfe_evo_ind]

    return bdfe_evo, prop_bdfe, dfe_anc





