import numpy as np

def init_sigma(N):
    """
    Initialize the spin configuration for the Sherrington-Kirkpatrick model.

    Parameters
    ----------
    N : int
        The number of spins.

    Returns
    -------
    numpy.ndarray
        The spin configuration.
    """
    return np.random.choice([-1, 1], N)


def compute_sigma_from_hist(sigma_0, hist, num_muts=None):
    """
    Compute sigma from the initial sigma and the flip history up to num_muts mutations.

    Parameters
    ----------
    sigma_0 : numpy.ndarray
        The initial spin configuration.
    hist : list of int
        The flip history.
    num_muts : int
        The number of mutations to consider.

    Returns
    -------
    numpy.ndarray
        The spin configuration after num_muts mutations.
    """
    sigma = sigma_0.copy()
    if num_muts is None:
        rel_hist = hist
    else:
        rel_hist = hist[:num_muts]
    for flip in rel_hist:
        sigma[flip] *= -1
    return sigma


def curate_sigma_list(sigma_0, hist, flips):
    """
    Curate the sigma list to have num_points elements.

    Parameters
    ----------
    sigma_0 : numpy.ndarray
        The initial spin configuration.
    hist : list of int
        The flip history.
    flips : list of int
        The points to curate.

    Returns
    -------
    list
        The curated list of spin configurations.
    """
    sigma_list = []
    for flip in flips:
        sigma_t = compute_sigma_from_hist(sigma_0, hist, flip)
        sigma_list.append(sigma_t)
    return sigma_list


def backward_propagate(dfe_evo: np.ndarray, dfe_anc: np.ndarray, beneficial=True):
    """
    Backward propagate the dfe from the initial day(anc) to the target day(evo),
    based on whether beneficial or deleterious mutations are selected.

    Parameters
    ----------
    dfe_evo : np. ndarray
        The dfe at the target day.
    dfe_anc : np. ndarray
        The dfe at the initial day.
    beneficial : bool, optional
        Whether to consider beneficial mutations. The default is True.

    Returns
    -------
    np. ndarray
        The propagated dfe.

    """
    bdfe_t = [(i, dfe_evo[i]) for i in range(len(dfe_evo)) if (dfe_evo[i] >= 0 if beneficial else dfe_evo[i] <= 0)]

    bdfe_t_inds = [x[0] for x in bdfe_t]
    bdfe_t_fits = [x[1] for x in bdfe_t]

    propagated_bdfe_t = [dfe_anc[i] for i in bdfe_t_inds]

    return bdfe_t_fits, propagated_bdfe_t


def forward_propagate(dfe_evo: np.ndarray, dfe_anc: np.ndarray, beneficial=True):
    """
    Forward propagate the dfe from the initial day(anc) to the target day(evo),
    based on whether beneficial or deleterious mutations are selected.

    Parameters
    ----------
    dfe_evo : np. ndarray
        The dfe at the target day.
    dfe_anc : np. ndarray
        The dfe at the initial day.
    beneficial : bool, optional
        Whether to consider beneficial mutations. The default is True.

    Returns
    -------
    np. ndarray
        The propagated dfe.
    """
    bdfe_0 = [(i, dfe_anc[i]) for i in range(len(dfe_anc)) if (dfe_anc[i] >= 0 if beneficial else dfe_anc[i] <= 0)]

    bdfe_0_inds = [x[0] for x in bdfe_0]
    bdfe_0_fits = [x[1] for x in bdfe_0]

    propagated_bdfe_0 = [dfe_evo[i] for i in bdfe_0_inds]

    return bdfe_0_fits, propagated_bdfe_0