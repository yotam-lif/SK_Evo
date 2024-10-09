import numpy as np

def init_alpha(N):
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
    sig_h = np.sqrt(1 - beta)
    return rng.normal(0.0, sig_h, N)


def init_J(N, random_state=None, beta=1.0, rho=1.0):
    """
    Initialize the coupling matrix for the Sherrington-Kirkpatrick model with sparsity.

    Parameters
    ----------
    N : int
        The number of spins.
    random_state : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.
    beta : float, optional
        Inverse temperature parameter.
    rho : float
        Fraction of non-zero elements in the coupling matrix (0 < rho ≤ 1).

    Returns
    -------
    numpy.ndarray
        The symmetric coupling matrix Jij with sparsity controlled by rho.
    """
    if not (0 < rho <= 1):
        raise ValueError("rho must be between 0 (exclusive) and 1 (inclusive).")
    rng = np.random.default_rng(random_state)
    sig_J = np.sqrt(beta / (N * rho) / 2)  # Adjusted standard deviation for sparsity
    # Initialize an empty upper triangular matrix (excluding diagonal)
    J_upper = np.zeros((N, N))
    # Total number of upper triangular elements excluding diagonal
    total_elements = N * (N - 1) // 2
    # Number of non-zero elements based on rho
    num_nonzero = int(np.floor(rho * total_elements))
    if num_nonzero == 0 and rho > 0:
        num_nonzero = 1  # Ensure at least one non-zero element if rho > 0
    # Get the indices for the upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(N, k=1)
    # Randomly select indices to assign non-zero Gaussian values
    selected_indices = rng.choice(total_elements, size=num_nonzero, replace=False)
    # Map the selected flat indices to row and column indices
    rows = triu_indices[0][selected_indices]
    cols = triu_indices[1][selected_indices]
    # Assign Gaussian-distributed values to the selected positions
    J_upper[rows, cols] = rng.normal(loc=0.0, scale=sig_J, size=num_nonzero)
    # Symmetrize the matrix to make Jij symmetric
    Jij = J_upper + J_upper.T

    return Jij


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

def calc_BDFE(alpha, h, J):
    """
    Calculate the Beneficial distribution of fitness effects.

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
    (numpy.ndarray, numpy.ndarray)
        The beneficial fitness effects and the indices of the beneficial mutations.
    """
    DFE = calc_DFE(alpha, h, J)
    return DFE[DFE > 0], np.where(DFE > 0)[0]


def calc_rank(alpha, h, J):
    """
    Calculate the rank of the spin configuration.

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
    int
        The rank of the spin configuration.
    """
    DFE = calc_DFE(alpha, h, J)
    return np.sum(DFE > 0)


def sswm_flip(alpha, his, Jijs):
    """
    Choose a spin to flip using probabilities of the sswm regime probabilities.

    Parameters
    ----------
    alpha : numpy.ndarray
        The spin configuration.
    his : numpy.ndarray
        The local fitness fields.
    Jijs : numpy.ndarray

    Returns
    -------
    int : The index of the spin to flip.
    """
    effects, indices = calc_BDFE(alpha, his, Jijs)
    effects /= np.sum(effects)
    return np.random.choice(indices, p=effects)


def glauber_flip(alpha, hi, Jij, beta=10):
    """
    Choose a spin to flip using the Glauber probabilities.

    Parameters
    ----------
    alpha : numpy.ndarray
        The spin configuration.
    hi : numpy.ndarray
        The local fitness fields.
    Jij : numpy.ndarray
        The coupling matrix.
    beta : float
        The inverse temperature.

    Returns
    -------
    numpy.ndarray
        The probability of flipping a spin.
    """
    kis = calc_kis(alpha, hi, Jij)
    ps = (1 - np.tanh(kis * beta)) / 2
    ps /= np.sum(ps)
    indices = range(len(alpha))
    return np.random.choice(indices, p=ps)

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
    return alpha @ (his + Jijs @ alpha) - F_off


def compute_fitness_delta_mutant(alpha, his, Jijs, k):
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

    return -2 * alpha[k] * (his[k] + Jijs[k] @ alpha)


def relax_SK(alpha, his, Jijs, ranks, sswm=True):
    """
    Relax the Sherrington-Kirkpatrick model with given parameters, saving alpha at specified ranks.

    Parameters
    ----------
    alpha : numpy.ndarray
        The initial spin configuration.
    his : numpy.ndarray
        The local fitness fields.
    Jijs : numpy.ndarray
        The coupling matrix.
    ranks : list or array-like
        The ranks at which to save the current alpha configuration.
    sswm : bool, optional
        Whether to use sswm_flip or glauber_flip for flipping spins.

    Returns
    -------
    (numpy.ndarray, list of numpy.ndarray or None)
        alpha: The final spin configuration.
        time_stamps: The list of alpha configurations saved at the specified ranks.
                     If a desired rank is not reached, `None` is saved for that rank.
    """
    # Ensure ranks are sorted in descending order
    ranks = sorted(ranks, reverse=True)
    time_stamps = [None] * len(ranks)  # Initialize with None
    current_index = 0
    rank = calc_rank(alpha, his, Jijs)

    while rank > 0 and current_index < len(ranks):
        # Check if the current rank matches the desired rank
        if rank <= ranks[current_index]:
            time_stamps[current_index] = alpha.copy()
            current_index += 1

        # Choose which flip method to use
        if sswm:
            flip_idx = sswm_flip(alpha, his, Jijs)
        else:
            # Since glauber_flip returns a single index, adjust accordingly
            flip_idx = glauber_flip(alpha, his, Jijs, beta=10)  # You can parameterize beta if needed

        # Flip the selected spin
        alpha[flip_idx] *= -1

        # Recalculate the rank after the flip
        rank = calc_rank(alpha, his, Jijs)
        print(f'Rank: {rank}')

    # Save the final configuration if the last rank is not reached
    time_stamps[current_index] = alpha.copy()

    return alpha, time_stamps
