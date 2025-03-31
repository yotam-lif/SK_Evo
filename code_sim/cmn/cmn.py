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


def compute_sigma_from_hist(sigma_0, hist, t=None):
    """
    Compute sigma from the initial sigma and the flip history up to flip number 't'.

    Parameters
    ----------
    sigma_0 : numpy.ndarray
        The initial spin configuration.
    hist : list of int
        The flip history - the indices of spins flipped, in order.
    t : int
        The flip number up to which to compute the spin configuration. The default is None.

    Returns
    -------
    numpy.ndarray
        The spin configuration after t mutations.
    """
    sigma = np.copy(sigma_0)
    if t is None:
        rel_hist = hist
    else:
        rel_hist = hist[:t]
    for flip in rel_hist:
        sigma[flip] *= -1
    return sigma


def curate_sigma_list(sigma_0, hist, ts):
    """
    Curate the sigma list to have num_points elements.

    Parameters
    ----------
    sigma_0 : numpy.ndarray
        The initial spin configuration.
    hist : list of int
        The flip history - the indices of spins flipped, in order.
    ts : list of int
        The indices of flips in 'hist' to recreate sigma at.

    Returns
    -------
    list
        The curated list of spin configurations.
    """
    sigma_list = []
    for t in ts:
        sigma_t = compute_sigma_from_hist(sigma_0, hist, t)
        sigma_list.append(sigma_t)
    return sigma_list