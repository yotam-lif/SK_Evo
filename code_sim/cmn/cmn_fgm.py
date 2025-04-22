import numpy as np

class Fisher:
    """
    Fisher Geometric Model with Gaussian mutation steps and SSWM relaxation.

    Attributes
    ----------
    n : int
        Dimensionality of phenotype space.
    S : numpy.ndarray
        Symmetric selection matrix (n x n), isotropic (alpha * I).
    delta : float
        Standard deviation for Gaussian mutation steps.
    dzs : numpy.ndarray
        Array of pre-sampled mutation steps of shape (m, n).
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    """

    def __init__(self, n, isotropic=True, alpha=1.0, delta=1.0, sigma_G=0.1, m=10**4, random_state=None):
        """
        Initialize model and pre-sample Gaussian mutation steps.

        Parameters
        ----------
        n : int
            Number of phenotypic traits (dimensions).
        isotropic : bool
            Always True for isotropic S = alpha * I.
        alpha : float
            Selection strength (for isotropic S).
        delta : float
            Std dev for Gaussian mutation steps.
        sigma_G : float
            Standard deviation for Gaussian entries in upper-triangular G (anisotropic).
        m : int
            Number of mutation steps to pre-sample.
        random_state : int or numpy.random.Generator, optional
            Seed or RNG for reproducibility.
        """
        self.n = int(n)
        self.delta = float(delta)
        self.m = int(m)
        # RNG setup
        if isinstance(random_state, (int, np.integer)):
            self.rng = np.random.default_rng(random_state)
        elif isinstance(random_state, np.random.Generator):
            self.rng = random_state
        else:
            self.rng = np.random.default_rng()
        # Build isotropic selection matrix S = alpha * I
        if isotropic:
            self.S = float(alpha) * np.eye(self.n)
        else:
            # anisotropic: draw upper-triangular G ~ N(0, sigma_G)
            G = self.rng.normal(loc=0.0, scale=float(sigma_G), size=(self.n, self.n))
            G = np.triu(G)
            S_raw = G + G.T
            S_sym = S_raw / 2.0
            # diagonalize S
            eigvals, eigvecs = np.linalg.eigh(S_sym)
            self.eigenvalues = eigvals
            self.eigenvectors = eigvecs
            # reconstruct for numerical symmetry
            self.S = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Pre-sample Gaussian mutation steps dz ~ N(0, delta^2 I)
        self.dzs = self.rng.normal(loc=0.0, scale=self.delta, size=(self.m, self.n))

    def compute_log_fitness(self, z):
        """
        Compute log-fitness: log w(z) = - z^T S z
        """
        z = np.asarray(z, dtype=float)
        return - float(z.T @ self.S @ z)

    def compute_fitness(self, z):
        """
        Compute fitness: w(z) = exp(log_fitness(z))
        """
        return float(np.exp(self.compute_log_fitness(z)))

    def compute_dfe(self, z):
        """
        Compute distribution of fitness effects at phenotype z.

        Parameters
        ----------
        z : array-like, shape (n,)
            Current phenotype vector.

        Returns
        -------
        dfe : numpy.ndarray, shape (m,)
            Fitness differences [w(z + dz_i) - w(z)] for each dz.
        """
        z = np.asarray(z, dtype=float)
        w0 = self.compute_fitness(z)
        # Evaluate fitness effect for each pre-sampled mutation
        return np.array([self.compute_fitness(z + dz) - w0 for dz in self.dzs])

    def compute_bdfe(self, dfe):
        """
        Extract beneficial fitness effects and their indices.

        Parameters
        ----------
        dfe : numpy.ndarray
            Distribution of fitness effects.

        Returns
        -------
        bdfe : numpy.ndarray
            Positive fitness effects.
        b_ind : numpy.ndarray
            Indices of beneficial mutations.
        """
        dfe = np.asarray(dfe, dtype=float)
        mask = dfe > 0
        return dfe[mask], np.nonzero(mask)[0]

    def sswm_choice(self, bdfe, b_ind):
        """
        Choose a mutation under Strong Selection Weak Mutation (SSWM):
        probability ∝ fitness effect.

        Parameters
        ----------
        bdfe : numpy.ndarray
            Beneficial fitness effects.
        b_ind : numpy.ndarray
            Indices of those effects.

        Returns
        -------
        int
            Index in dzs of chosen mutation.
        """
        bdfe = np.asarray(bdfe, dtype=float)
        total = bdfe.sum()
        if total > 0:
            probs = bdfe / total
        else:
            probs = np.ones_like(bdfe) / len(bdfe)
        return int(self.rng.choice(b_ind, p=probs))

    def relax(self, z_init, max_steps=1000):
        """
        Perform an adaptive walk using SSWM, moving towards the optimum.

        Parameters
        ----------
        z_init : array-like, shape (n,)
            Starting phenotype vector.
        max_steps : int
            Maximum number of substitutions.

        Returns
        -------
        flips : list of int
            Indices of chosen dzs at each substitution.
        traj : list of numpy.ndarray
            Phenotype vectors after each substitution.
        """
        z = np.asarray(z_init, dtype=float).copy()
        traj = [z.copy()]
        flips = []
        for _ in range(max_steps):
            dfe = self.compute_dfe(z)
            bdfe, b_ind = self.compute_bdfe(dfe)
            if len(b_ind) == 0:
                break
            choice = self.sswm_choice(bdfe, b_ind)
            flips.append(choice)
            z += self.dzs[choice]
            traj.append(z.copy())
        return flips, traj