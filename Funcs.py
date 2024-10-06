import numpy as np

def init_J(N, random_state=None):
    """
    Initialize the coupling matrix for the Sherrington-Kirkpatrick model.

    Parameters
    ----------
    N : int
        The number of spins.
    random_state : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    numpy.ndarray
        The coupling matrix.
    """
    rng = np.random.default_rng(random_state)
    J = rng.normal(0.0, 1 / N, (N, N))
    J_upper = np.triu(J, 1)
    J = J_upper + J_upper.T
    np.fill_diagonal(J, 0.0)
    return J

def sswm_prob(kis):
    """
    Calculate the probability of flipping a spin based on the energy delta of the system.

    Parameters
    ----------
    kis : numpy.ndarray
        The energy delta of the system.

    Returns
    -------
    numpy.ndarray
        The probability of flipping a spin.
    """
    ps = np.where(kis < 0, -1 * kis, 0)
    return ps / np.sum(ps)

def glauber_prob(kis, beta=10):
    """
    Calculate the probability of flipping a spin based on the energy delta of the system.

    Parameters
    ----------
    kis : numpy.ndarray
        The energy delta of the system.
    beta : float
        The inverse temperature.

    Returns
    -------
    numpy.ndarray
        The probability of flipping a spin.
    """
    return (1 - np.tanh(kis * beta)) / 2

def calc_rank(alpha, Jij, his=None):
    """
    Calculate the rank of the spin configuration.

    Parameters
    ----------
    alpha : numpy.ndarray
        The spin configuration.
    Jij : numpy.ndarray
        The coupling matrix.
    his : numpy.ndarray, optional

    Returns
    -------
    int
        The rank of the spin configuration.
    """
    if his is None:
        his = alpha @ Jij
    kis = alpha * his
    # return how many negative elements are in kis, i.e. home many spins are anti-aligned with local field
    return np.sum(kis < 0)

def relax_SK(alpha, Jij, prob_func, random_state=None):
    """
    Relax the Sherrington-Kirkpatrick model with given parameters.

    Parameters
    ----------
    alpha : numpy.ndarray
        The initial spin configuration.
    Jij : numpy.ndarray
        The coupling matrix.
    prob_func : function
        The function to calculate the probability of flipping a spin based on the energy delta of the system.
    max_iterations : int, optional
        Maximum number of iterations to prevent infinite loops.
    random_state : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    numpy.ndarray
        The final spin configuration.
    """
    rng = np.random.default_rng(random_state)
    his = alpha @ Jij
    rank = calc_rank(alpha, Jij, his)

    while rank > 0:
        kis = -2 * alpha * his
        ps = prob_func(kis)
        flip_idx = rng.choice(len(alpha), p=ps)
        alpha[flip_idx] *= -1
        his = alpha @ Jij
        rank = calc_rank(alpha, Jij, his)
        print(f'Rank: {rank}')

    return alpha
