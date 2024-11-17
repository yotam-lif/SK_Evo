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
        Divide by 2 because every term appears twice in symmetric case.
    """
    return h + 0.5 * J @ alpha


def calc_energies(alpha, h, J):
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
        The normalized distribution of fitness effects.
    """
    return -2 * calc_energies(alpha, h, J)


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
    r = calc_rank(alpha, h, J)
    BDFE, b_ind = DFE[DFE >= 0], np.where(DFE > 0)[0]
    # Normalize the beneficial effects
    # BDFE /= np.sum(BDFE)
    return BDFE, b_ind


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
    eis = calc_energies(alpha, hi, Jij)
    ps = (1 - np.tanh(eis * beta)) / 2
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
    Divide by 2 because every term appears twice in symmetric case.
    """
    return alpha @ (his + 0.5 * Jijs @ alpha) - F_off


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
    Divide by 2 because every term appears twice in symmetric case.
    """

    return -2 * alpha[k] * (his[k] + 0.5 * Jijs[k] @ alpha)


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


def relax_sk_flips(alpha, his, Jijs, flips, sswm=True):
    """
    Relax the Sherrington-Kirkpatrick model with given parameters, saving alpha at specified flips.
    Parameters
    ----------
    alpha: numpy.ndarray
    his: numpy.ndarray
    Jijs: numpy.ndarray
    flips: list or array-like
    sswm: bool, optional

    Returns
    -------
    numpy.ndarray
        The final spin configuration, saved alphas.
    """
    saved_alphas = []
    total_flips = 0
    rank = calc_rank(alpha, his, Jijs)

    while total_flips < flips[-1] and rank > 0:

        if total_flips in flips:
            saved_alphas.append(alpha.copy())

        flip_idx = sswm_flip(alpha, his, Jijs) if sswm else glauber_flip(alpha, his, Jijs, beta=10)
        alpha[flip_idx] *= -1
        total_flips += 1
        rank = calc_rank(alpha, his, Jijs)

    if rank == 0:
        saved_alphas.append(alpha.copy())
        print("Not all flips reached, rank is 0")

    return alpha, saved_alphas


def relax_sk_ranks(alpha, his, Jijs, num_ranks, fin_rank=0, sswm=True):
    """
    Relax the Sherrington-Kirkpatrick model with given parameters, saving alpha at specified ranks.
    Parameters
    ----------
    alpha: numpy.ndarray
    his: numpy.ndarray
    Jijs: numpy.ndarray
    fin_rank: int
    num_ranks: int
    sswm: bool, optional

    Returns
    -------
    numpy.ndarray, list
        The final spin configuration, saved alphas.
    """
    saved_alphas = []
    rank = calc_rank(alpha, his, Jijs)
    ranks = sorted(np.linspace(rank, fin_rank, num_ranks, dtype=int), reverse=True)

    while rank > fin_rank:

        if rank in ranks:
            saved_alphas.append(alpha.copy())

        flip_idx = sswm_flip(alpha, his, Jijs) if sswm else glauber_flip(alpha, his, Jijs, beta=10)
        alpha[flip_idx] *= -1
        rank = calc_rank(alpha, his, Jijs)

    # Save the final alpha
    saved_alphas.append(alpha.copy())
    return alpha, saved_alphas, ranks


def relax_sk(alpha, his, Jijs, sswm=True):
    """
    Relax the Sherrington-Kirkpatrick model with given parameters.
    Parameters
    ----------
    alpha: numpy.ndarray
    his: numpy.ndarray
    Jijs: numpy.ndarray
    sswm: bool, optional

    Returns
    -------
    list
        The mutation sequence.
    """
    flip_sequence = []
    rank = calc_rank(alpha, his, Jijs)

    while rank > 0:
        flip_idx = sswm_flip(alpha, his, Jijs) if sswm else glauber_flip(alpha, his, Jijs, beta=10)
        alpha[flip_idx] *= -1
        flip_sequence.append(flip_idx)
        rank = calc_rank(alpha, his, Jijs)

    return flip_sequence


def compute_alpha_from_hist(alpha_0, hist, num_muts):
    """
    Compute alpha from the initial alpha and the flip history up to num_muts mutations.

    Parameters
    ----------
    alpha_0 : numpy.ndarray
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
    alpha = alpha_0.copy()
    rel_hist = hist[:num_muts]
    for flip in rel_hist:
        alpha[flip] *= -1
    return alpha
