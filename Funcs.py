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

def calc_F_off(alpha_init, his, Jijs):
    """
    Calculate the fitness offset for the given configuration.

    Parameters
    ----------
    alpha_init : numpy.ndarray
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
    return compute_fit_slow(alpha_init, his, Jijs) - 1

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
    Relax the Sherrington-Kirkpatrick model with given parameters, saving alpha at specified ranks,
    and recording the cumulative number of flips (mutations) up to each saved rank.

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
    (numpy.ndarray, list of numpy.ndarray or None, list of int)
        alpha: The final spin configuration.
        time_stamps: The list of alpha configurations saved at the specified ranks.
                     If a desired rank is not reached, `None` is saved for that rank.
        flips: List of cumulative number of flips up to each saved alpha configuration.
    """
    # Ensure ranks are sorted in descending order
    ranks = sorted(ranks, reverse=True)
    time_stamps = [None] * len(ranks)  # Initialize with None
    flips = [None] * len(ranks)  # Initialize flips list corresponding to each saved configuration
    current_index = 0
    rank = calc_rank(alpha, his, Jijs)
    total_flips = 0  # Initialize cumulative flip counter

    while rank > 0 and current_index < len(ranks):
        # Check if the current rank matches or falls below the desired rank
        if rank <= ranks[current_index]:
            time_stamps[current_index] = alpha.copy()
            flips[current_index] = total_flips
            current_index += 1

            # If we've reached all desired ranks, break the loop
            if current_index >= len(ranks):
                break

        # Choose which flip method to use
        if sswm:
            flip_idx = sswm_flip(alpha, his, Jijs)
        else:
            # Since glauber_flip returns a single index, adjust accordingly
            flip_idx = glauber_flip(alpha, his, Jijs, beta=10)  # You can parameterize beta if needed

        # Flip the selected spin
        alpha[flip_idx] *= -1
        total_flips += 1  # Increment the total flips counter

        # Recalculate the rank after the flip
        rank = calc_rank(alpha, his, Jijs)
        # Uncomment the following line if you want to see the rank progression
        # print(f'Rank: {rank}')

    # Save the final configuration and flips if the last rank was not reached
    if current_index < len(ranks):
        time_stamps[current_index] = alpha.copy()
        flips[current_index] = total_flips

    return alpha, time_stamps, flips

def backward_propagate(dfe_evo: np. ndarray, dfe_anc: np. ndarray, beneficial=True):
    """
    Backward propagate the DFE from the initial day(anc) to the target day(evo),
    based on whether beneficial or deleterious mutations are selected.

    Parameters
    ----------
    dfe_evo : np. ndarray
        The DFE at the target day.
    dfe_anc : np. ndarray
        The DFE at the initial day.
    beneficial : bool, optional
        Whether to consider beneficial mutations. The default is True.

    Returns
    -------
    np. ndarray
        The propagated DFE.

    """
    bdfe_t = [(i, dfe_evo[i]) for i in range(len(dfe_evo)) if (dfe_evo[i] >= 0 if beneficial else dfe_evo[i] <= 0)]

    bdfe_t_inds = [x[0] for x in bdfe_t]
    bdfe_t_fits = [x[1] for x in bdfe_t]

    propagated_bdfe_t = [dfe_anc[i] for i in bdfe_t_inds]

    return bdfe_t_fits, propagated_bdfe_t


def forward_propagate(dfe_evo: np. ndarray, dfe_anc: np. ndarray, beneficial=True):
    """
    Forward propagate the DFE from the initial day(anc) to the target day(evo),
    based on whether beneficial or deleterious mutations are selected.

    Parameters
    ----------
    dfe_evo : np. ndarray
        The DFE at the target day.
    dfe_anc : np. ndarray
        The DFE at the initial day.
    beneficial : bool, optional
        Whether to consider beneficial mutations. The default is True.

    Returns
    -------
    np. ndarray
        The propagated DFE.
    """
    bdfe_0 = [(i, dfe_anc[i]) for i in range(len(dfe_anc)) if (dfe_anc[i] >= 0 if beneficial else dfe_anc[i] <= 0)]

    bdfe_0_inds = [x[0] for x in bdfe_0]
    bdfe_0_fits = [x[1] for x in bdfe_0]

    propagated_bdfe_0 = [dfe_evo[i] for i in bdfe_0_inds]

    return bdfe_0_fits, propagated_bdfe_0
