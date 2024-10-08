import numpy as np

def init_J(N, random_state=None, beta=1.0, rho=1.0):
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
    sig_J_sq = beta / (N * rho) / 2
    J = rng.normal(0.0, sig_J_sq, (N, N))
    J_upper = np.triu(J, 1)
    J = J_upper + J_upper.T
    np.fill_diagonal(J, 0.0)
    return J

def init_h(N, random_state=None, beta=1.0):
    """
    Initialize the external fields for the Sherrington-Kirkpatrick model.

    Parameters
    ----------
    N : int
        The number of spins.
    random_state : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    numpy.ndarray
        The external fields.
    """
    rng = np.random.default_rng(random_state)
    sig_h_sq = 1 - beta
    return rng.normal(0.0, np.sqrt(sig_h_sq), N)

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
    ps = np.where(kis < 0, np.abs(kis), 0)
    total = ps.sum()
    if total > 0:
        return ps / total
    else:
        # Return uniform probabilities if no spins qualify
        return np.ones_like(ps) / len(ps)


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


def calc_basic_lfs(alpha, h, J):
    """
    Calculate the local fields for the Sherrington-Kirkpatrick model.

    Parameters
    ----------
    alpha : numpy.ndarray
        The spin configuration.
    h : numpy.ndarray
        The external fields.
    J : numpy.ndarray
        The coupling matrix.

    Returns
    -------
    numpy.ndarray
        The local fields.
    """
    return h + J @ alpha

def calc_kis(alpha, h, J):
    """
    Calculate the energy delta of the system.

    Parameters
    ----------
    alpha : numpy.ndarray
        The spin configuration.
    h : numpy.ndarray
        The external fields.
    J : numpy.ndarray
        The coupling matrix.

    Returns
    -------
    numpy.ndarray
        The kis of the system.
    """
    return alpha * calc_basic_lfs(alpha, h, J)

def calc_DFE(alpha, h, J):
    """
    Calculate the distribution of fitness effects.

    Parameters
    ----------
    alpha : numpy.ndarray
        The spin configuration.
    h : numpy.ndarray
        The external fields.
    J : numpy.ndarray
        The coupling matrix.

    Returns
    -------
    numpy.ndarray
        The distribution of fitness effects.
    """
    return -2 * calc_kis(alpha, h, J)



def calc_rank(alpha, h, J):
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
    kis = calc_kis(alpha, h, J)
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
        kis = alpha * his
        ps = prob_func(kis)
        flip_idx = rng.choice(len(alpha), p=ps)
        alpha[flip_idx] *= -1
        his = alpha @ Jij
        rank = calc_rank(alpha, Jij, his)
        # print(f'Rank: {rank}')

    return alpha

def calc_DFE(alpha, Jij):
    """
    Calculate the distribution of local field energies.

    Parameters
    ----------
    alpha : numpy.ndarray
        The spin configuration.
    Jij : numpy.ndarray
        The coupling matrix.

    Returns
    -------
    numpy.ndarray
        The distribution of fitness effects.
    """
    his = alpha @ Jij
    kis = alpha * his
    return -2 * kis

def compute_fit_slow(alpha, his, Jijs, F_off=0.0):
    """
    Compute the fitness of the genome configuration alpha using full slow computation.

    Parameters:
    alpha (np.ndarray): The genome configuration (vector of -1 or 1).
    his (np.ndarray): The vector of site-specific contributions to fitness.
    Jijs (np.ndarray): The interaction matrix between genome sites.
    F_off (float): The fitness offset, defaults to 0.

    Returns:
    float: The fitness value for the configuration alpha.
    """
    return alpha @ his + alpha @ Jijs @ alpha - F_off


def compute_fitness_delta_mutant(alpha, hi, f_i, k):
    """
    Compute the fitness change for a mutant at site k.

    Parameters:
    alpha (np.ndarray): The genome configuration.
    hi (np.ndarray): The vector of site-specific fitness contributions.
    f_i (np.ndarray): The local fitness fields.
    k (int): The index of the mutation site.

    Returns:
    float: The change in fitness caused by a mutation at site k.
    """
    return -2 * alpha[k] * (hi[k] + f_i[k])
