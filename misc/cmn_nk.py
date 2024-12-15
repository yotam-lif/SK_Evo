import numpy as np

class NK:
    """
    The NK model.
    N: Number of loci.
    K: Each locus's fitness depends on itself and its K neighbors.
    """

    def __init__(self, N, K, mean=0.0, std=1.0, seed=None):
        """
        Initialize the NK model.

        Parameters
        ----------
        N : int
            Number of loci.
        K : int
            Number of neighbors per locus.
        mean : float
            Mean of the Gaussian distribution used for fitness draws.
        std : float
            Standard deviation of the Gaussian distribution used for fitness draws.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.N = N
        self.K = K
        self.mean = mean
        self.std = std

        if seed is not None:
            np.random.seed(seed)

        # contributions[i] is a dictionary where:
        # key: tuple of states (S_i, S_j1, ..., S_jK)
        # value: fitness value drawn from Gaussian
        self.fis = [dict() for _ in range(N)]

    def compute_fitness(self, sigma, f_off=0.0):
        """
        Compute the total fitness of a given configuration.
        The fitness is the mean of f_i over all loci i.

        Parameters
        ----------
        sigma : numpy.ndarray
            Array of locus states, typically -1 or +1.
        f_off : float, optional
            Fitness offset to be subtracted from the total fitness. Default is 0.0.

        Returns
        -------
        float
            The total fitness of the configuration.
        """
        fit_sum = 0.0

        for i in range(self.N):
            # Identify the pattern: locus i and its K neighbors (circular)
            # We'll consider the next K loci in a circular fashion.
            indices = [(i + offset) % self.N for offset in range(self.K + 1)]
            kclique_i = tuple(sigma[idx] for idx in indices)
            # Check if we already have a fitness value for this pattern
            if kclique_i not in self.fis[i]:
                # Draw from a Gaussian distribution and store
                self.fis[i][kclique_i] = np.random.normal(self.mean, self.std)
            # Add the contribution for this pattern
            fit_sum += self.fis[i][kclique_i]

        # The total fitness is the average of all f_i
        return fit_sum / self.N - f_off


def compute_dfe(sigma, nk, f_off=0.0):
    """
    Compute the distribution of fitness effects (DFE) for a given configuration.

    Parameters
    ----------
    sigma : numpy.ndarray
        Array of locus states, typically -1 or +1.
    nk : NK
        An instance of the NK model.
    f_off : float, optional
        Fitness offset to be subtracted from the total fitness. Default is 0.0.

    Returns
    -------
    numpy.ndarray
        The DFE for the given configuration.
    """
    dfe = np.zeros(nk.N)
    curr_fit = nk.compute_fitness(sigma, f_off)
    for i in range(nk.N):
        sigma_prime = sigma.copy()
        sigma_prime[i] = -sigma_prime[i]
        dfe[i] = nk.compute_fitness(sigma_prime, f_off) - curr_fit
    return dfe

def compute_bdfe(sigma, nk, f_off=0.0):
    """
    Compute the beneficial distribution of fitness effects (bDFE) for a given configuration.

    Parameters
    ----------
    sigma : numpy.ndarray
        Array of locus states, typically -1 or +1.
    nk : NK
        An instance of the NK model.
    f_off : float, optional
        Fitness offset to be subtracted from the total fitness. Default is 0.0.

    Returns
    -------
    numpy.ndarray
        The bDFE for the given configuration.
    numpy.ndarray
        Indices of beneficial mutations.
    """
    dfe = compute_dfe(sigma, nk, f_off)
    bdfe = dfe[dfe >= 0]
    b_ind = np.where(dfe >= 0)[0]
    return bdfe, b_ind

def compute_rank(sigma, nk, f_off=0.0):
    """
    Compute the rank of a given configuration.
    The rank is the number of beneficial mutations.

    Parameters
    ----------
    sigma : numpy.ndarray
        Array of locus states, typically -1 or +1.
    nk : NK
        An instance of the NK model.
    f_off : float, optional
        Fitness offset to be subtracted from the total fitness. Default is 0.0.

    Returns
    -------
    int
        The rank of the configuration.
    """
    dfe = compute_dfe(sigma, nk, f_off)
    return np.sum(dfe >= 0)

def sswm_choice(sigma, nk, f_off=0.0):
    """
    Select a site for mutation using the Strong Selection Weak Mutation (SSWM) model.

    Parameters
    ----------
    sigma : numpy.ndarray
        Array of locus states, typically -1 or +1.
    nk : NK
        An instance of the NK model.
    f_off : float, optional
        Fitness offset to be subtracted from the total fitness. Default is 0.0.

    Returns
    -------
    int
        The index of the selected site for mutation.
    """
    bdfe, b_ind = compute_bdfe(sigma, nk, f_off)
    bdfe /= np.sum(bdfe)
    return np.random.choice(b_ind, p=bdfe)

def relax_nk(sigma_init, nk, f_off=0.0):
    """
    Relax a given configuration using the NK model.

    Parameters
    ----------
    sigma_init : numpy.ndarray
        Initial array of locus states, typically -1 or +1.
    nk : NK
        An instance of the NK model.
    f_off : float, optional
        Fitness offset to be subtracted from the total fitness. Default is 0.0.

    Returns
    -------
    numpy.ndarray
        The relaxed configuration.
    """
    sigma = sigma_init.copy()
    rank = compute_rank(sigma, nk, f_off)
    flip_hist = []
    while rank > 0:
        i = sswm_choice(sigma, nk, f_off)
        flip_hist.append(i)
        sigma[i] = -sigma[i]
        rank = compute_rank(sigma, nk, f_off)
    return flip_hist