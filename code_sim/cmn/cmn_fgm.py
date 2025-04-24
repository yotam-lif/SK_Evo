import numpy as np

class Fisher:
    """
    Fisher Geometric Model with Gaussian mutation steps and SSWM relaxation.

    In this version, the selection matrix S is represented only by its eigenvalues
    (i.e., working in the diagonal basis). Both isotropic and anisotropic cases
    are supported through the eigenvalue spectrum.

    Attributes
    ----------
    n : int
        Dimensionality of phenotype space.
    eigenvalues : numpy.ndarray
        Array of length n giving the diagonal entries of S.
    delta : float
        Standard deviation for Gaussian mutation steps.
    dzs : numpy.ndarray
        Array of pre-sampled mutation steps of shape (m, n).
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    """

    def __init__(self, n, delta, isotropic=True, sigma=1.0, m=10**4, random_state=None):
        """
        Initialize the model in the diagonal basis.

        Parameters
        ----------
        n : int
            Number of phenotypic traits (dimensions).
        isotropic : bool
            If True, all eigenvalues of S = 1; otherwise drawn from semicircle.
        delta : float
            Std dev for Gaussian mutation steps.
        sigma : float
            Scale parameter (radius/2) for semicircle eigenvalue distribution.
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

        # Set eigenvalues of S
        if isotropic:
            # isotropic: all eigenvalues = 1
            self.eigenvalues = np.ones(self.n)
        else:
            # anisotropic: eigenvalues ~ semicircle on [-2*sigma,2*sigma]
            self.eigenvalues = self._sample_semicircle(self.n, sigma)

        # Pre-sample Gaussian mutation steps dz ~ N(0, delta^2 I)
        self.dzs = self.rng.normal(loc=0.0, scale=self.delta, size=(self.m, self.n))

    def _sample_semicircle(self, n, sigma):
        """
        Sample n values from the semicircle distribution on [-2*sigma, 2*sigma]:
        density f(x) ∝ sqrt(4*sigma^2 - x^2) using rejection sampling.
        """
        radius = 2.0 * sigma
        samples = []
        while len(samples) < n:
            x = self.rng.uniform(-radius, radius)
            accept_prob = np.sqrt(radius**2 - x**2) / radius
            if self.rng.uniform(0.0, 1.0) < accept_prob:
                samples.append(x)
        return np.array(samples)

    def compute_log_fitness(self, z):
        """
        Compute log-fitness in diagonal basis:
        log w(z) = - z^T S z = - sum_i eigenvalues[i] * z[i]^2
        """
        z = np.asarray(z, dtype=float)
        return - float(np.sum(self.eigenvalues * z**2))

    def compute_fitness(self, z):
        """
        Compute fitness: w(z) = exp(log_fitness(z)).
        """
        return float(np.exp(self.compute_log_fitness(z)))

    def compute_dfe(self, z):
        """
        Compute distribution of fitness effects at phenotype z.
        Returns array of w(z + dz_i) - w(z) for each pre-sampled dz.
        """
        z = np.asarray(z, dtype=float)
        w0 = self.compute_fitness(z)
        return np.array([self.compute_fitness(z + dz) - w0 for dz in self.dzs])

    def compute_bdfe(self, dfe):
        """
        Extract beneficial fitness effects and their indices from dfe array.
        """
        dfe = np.asarray(dfe, dtype=float)
        mask = dfe > 0
        return dfe[mask], np.nonzero(mask)[0]

    def sswm_choice(self, bdfe, b_ind):
        """
        Choose a substitution under SSWM: probability ∝ fitness effect.
        """
        bdfe = np.asarray(bdfe, dtype=float)
        total = bdfe.sum()
        if total > 0:
            probs = bdfe / bdfe.sum()
        else:
            probs = bdfe / len(bdfe)
        return int(self.rng.choice(b_ind, p=probs))

    def relax(self, z_init, max_steps=1000):
        """
        Perform an adaptive walk using SSWM.
        Returns list of chosen mutation indices and trajectory of z.
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
            self.dzs[choice] = -1 * self.dzs[choice]
            traj.append(z.copy())
        return flips, traj
