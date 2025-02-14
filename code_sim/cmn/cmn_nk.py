import numpy as np
from collections import defaultdict


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

        # fis[i] is a dictionary where:
        # key: tuple of states (S_i, S_j1, ..., S_jK)
        # value: fitness value drawn from Gaussian
        # defaultdict takes care of sampling a new RV if the key is not in dictionary i
        self.fis = [defaultdict(lambda: np.random.normal(self.mean, self.std)) for _ in range(N)]
        # Precompute neighbor indices (circular)
        self.neighbor_indices = [
            [(i + offset) % self.N for offset in range(self.K + 1)] for i in range(self.N)
        ]
        # Precompute dependents for each locus, where dependents are the indices which fitness is affected by locus i
        self.dependents = [[] for _ in range(self.N)]
        for j in range(self.N):
            for i in self.neighbor_indices[j]:
                self.dependents[i].append(j)

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
            indices = self.neighbor_indices[i]
            kclique_i = tuple(sigma[idx] for idx in indices)
            # defaultdict takes care of the case where kclique_i is not in contributions[i]
            fit_sum += self.fis[i][kclique_i]
        # The total fitness is the average of all f_i
        return fit_sum / self.N - f_off

    def get_fis(self):
        """
        Get the fitness values for all loci and patterns.

        Returns
        -------
        list:
            List of dictionaries where each dictionary contains the fitness values for a locus.
        """
        return self.fis

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
        # avoid excess copying of sigma prime by switching back each flip
        sigma[i] = -sigma[i]
        dfe[i] = nk.compute_fitness(sigma, f_off) - curr_fit
        sigma[i] = -sigma[i]
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
    sigma = np.copy(sigma_init)
    rank = compute_rank(sigma, nk, f_off)
    flip_hist = []
    while rank > 0:
        i = sswm_choice(sigma, nk, f_off)
        flip_hist.append(i)
        sigma[i] = -sigma[i]
        rank = compute_rank(sigma, nk, f_off)
    return flip_hist, nk
