import numpy as np


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
    sig_J = np.sqrt(beta / (N * rho))  # Adjusted standard deviation for sparsity
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


def calc_basic_lfs(sigma, h, J):
    """
    Calculate the local fields for the Sherrington-Kirkpatrick model.

    Parameters
    ----------
    sigma : numpy.ndarray
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
    return h + J @ sigma


def calc_energies(sigma, h, J):
    """
    Calculate the energy delta of the system.

    Parameters
    ----------
    sigma : numpy.ndarray
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
    return sigma * calc_basic_lfs(sigma, h, J)


def calc_DFE(sigma, h, J):
    """
    Calculate the distribution of fitness effects.

    Parameters
    ----------
    sigma : numpy.ndarray
        The spin configuration.
    h : numpy.ndarray
        The external fields.
    J : numpy.ndarray
        The coupling matrix.

    Returns
    -------
    numpy.ndarray
        The normalized distribution of fitness effects.
    """
    return -2 * sigma * calc_basic_lfs(sigma, h, J)


def calc_BDFE(sigma, h, J):
    """
    Calculate the Beneficial distribution of fitness effects.

    Parameters
    ----------
    sigma : numpy.ndarray
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
    DFE = calc_DFE(sigma, h, J)
    BDFE, b_ind = DFE[DFE >= 0], np.where(DFE > 0)[0]
    # Normalize the beneficial effects
    # BDFE /= np.sum(BDFE)
    return BDFE, b_ind


def calc_rank(sigma, h, J):
    """
    Calculate the rank of the spin configuration.

    Parameters
    ----------
    sigma : numpy.ndarray
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
    DFE = calc_DFE(sigma, h, J)
    return np.sum(DFE > 0)


def sswm_flip(sigma, his, Jijs):
    """
    Choose a spin to flip using probabilities of the sswm regime probabilities.

    Parameters
    ----------
    sigma : numpy.ndarray
        The spin configuration.
    his : numpy.ndarray
        The local fitness fields.
    Jijs : numpy.ndarray

    Returns
    -------
    int : The index of the spin to flip.
    """
    effects, indices = calc_BDFE(sigma, his, Jijs)
    effects /= np.sum(effects)
    return np.random.choice(indices, p=effects)


def glauber_flip(sigma, hi, Jij, beta=10):
    """
    Choose a spin to flip using the Glauber probabilities.

    Parameters
    ----------
    sigma : numpy.ndarray
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
    eis = calc_energies(sigma, hi, Jij)
    ps = (1 - np.tanh(eis * beta)) / 2
    ps /= np.sum(ps)
    indices = range(len(sigma))
    return np.random.choice(indices, p=ps)


def calc_F_off(sigma_init, his, Jijs):
    """
    Calculate the fitness offset for the given configuration.

    Parameters
    ----------
    sigma_init : numpy.ndarray
        The initial spin configuration.
    his : numpy.ndarray
        The local fitness fields.
    Jijs : numpy.ndarray
        The coupling matrix.

    Returns
    -------
    float
        The fitness offset.
    """
    return compute_fit_slow(sigma_init, his, Jijs) - 1


def compute_fit_slow(sigma, his, Jijs, F_off=0.0):
    """
    Compute the fitness of the genome configuration sigma using full slow computation.

    Parameters:
    sigma (np.ndarray): The genome configuration (vector of -1 or 1).
    his (np.ndarray): The vector of site-specific contributions to fitness.
    Jijs (np.ndarray): The interaction matrix between genome sites.
    F_off (float): The fitness offset, defaults to 0.

    Returns:
    float: The fitness value for the configuration sigma.
    Divide by 2 because every term appears twice in symmetric case.
    """
    return sigma @ (his + 0.5 * Jijs @ sigma) - F_off


def compute_fitness_delta_mutant(sigma, his, Jijs, k):
    """
    Compute the fitness change for a mutant at site k.

    Parameters:
    sigma (np.ndarray): The genome configuration.
    hi (np.ndarray): The vector of site-specific fitness contributions.
    f_i (np.ndarray): The local fitness fields.
    k (int): The index of the mutation site.

    Returns:
    float: The change in fitness caused by a mutation at site k.
    Divide by 2 because every term appears twice in symmetric case.
    """

    return -2 * sigma[k] * (his[k] + Jijs[k] @ sigma)


def relax_sk_flips(sigma, his, Jijs, flips, sswm=True):
    """
    Relax the Sherrington-Kirkpatrick model with given parameters, saving sigma at specified flips.
    Parameters
    ----------
    sigma: numpy.ndarray
    his: numpy.ndarray
    Jijs: numpy.ndarray
    flips: list or array-like
    sswm: bool, optional

    Returns
    -------
    numpy.ndarray
        The final spin configuration, saved sigmas.
    """
    saved_sigmas = []
    total_flips = 0
    rank = calc_rank(sigma, his, Jijs)

    while total_flips < flips[-1] and rank > 0:

        if total_flips in flips:
            saved_sigmas.append(sigma.copy())

        flip_idx = sswm_flip(sigma, his, Jijs) if sswm else glauber_flip(sigma, his, Jijs, beta=10)
        sigma[flip_idx] *= -1
        total_flips += 1
        rank = calc_rank(sigma, his, Jijs)

    if rank == 0:
        saved_sigmas.append(sigma.copy())
        print("Not all flips reached, rank is 0")

    return sigma, saved_sigmas


def relax_sk_ranks(sigma, his, Jijs, num_ranks, fin_rank=0, sswm=True):
    """
    Relax the Sherrington-Kirkpatrick model with given parameters, saving sigma at specified ranks.
    Parameters
    ----------
    sigma: numpy.ndarray
    his: numpy.ndarray
    Jijs: numpy.ndarray
    fin_rank: int
    num_ranks: int
    sswm: bool, optional

    Returns
    -------
    numpy.ndarray, list
        The final spin configuration, saved sigmas.
    """
    saved_sigmas = []
    rank = calc_rank(sigma, his, Jijs)
    ranks = sorted(np.linspace(rank, fin_rank, num_ranks, dtype=int), reverse=True)

    while rank > fin_rank:

        if rank in ranks:
            saved_sigmas.append(sigma.copy())

        flip_idx = sswm_flip(sigma, his, Jijs) if sswm else glauber_flip(sigma, his, Jijs, beta=10)
        sigma[flip_idx] *= -1
        rank = calc_rank(sigma, his, Jijs)

    # Save the final sigma
    saved_sigmas.append(sigma.copy())
    return sigma, saved_sigmas, ranks


def relax_sk(sigma, his, Jijs, sswm=True):
    """
    Relax the Sherrington-Kirkpatrick model with given parameters.
    Parameters
    ----------
    sigma: numpy.ndarray
    his: numpy.ndarray
    Jijs: numpy.ndarray
    sswm: bool, optional

    Returns
    -------
    list
        The mutation sequence.
    """
    flip_sequence = []
    rank = calc_rank(sigma, his, Jijs)

    while rank > 0:
        flip_idx = sswm_flip(sigma, his, Jijs) if sswm else glauber_flip(sigma, his, Jijs, beta=10)
        sigma[flip_idx] *= -1
        flip_sequence.append(flip_idx)
        rank = calc_rank(sigma, his, Jijs)

    return flip_sequence


