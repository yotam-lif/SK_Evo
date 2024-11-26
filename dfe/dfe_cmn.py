# Import the Funcs module
from misc import cmn

def gen_bdfes(N, beta, rho, num_points, num_repeats):
    """
    Generate BDFE histograms.
    Parameters
    ----------
    N: int
    beta: float
    rho: float
    num_points: int
    num_repeats: int

    Returns
    -------
    bdfes: list
    """
    bdfes = [[] for _ in range(num_points)]

    for repeat in range(num_repeats):
        print(f"\nStarting repeat {repeat + 1} of {num_repeats}...")
        # Initialize spin configuration
        alpha_initial = cmn.init_alpha(N)

        # Initialize external fields
        h = cmn.init_h(N, beta=beta)

        # Initialize coupling matrix with sparsity
        J = cmn.init_J(N, beta=beta, rho=rho)

        # Relax the system using sswm_flip (sswm=True)
        flip_seq = cmn.relax_sk(alpha_initial.copy(), h, J, sswm=True)
        alphas, _ = cmn.curate_alpha_list(alpha_initial, flip_seq, num_points)
        for i, alpha in enumerate(alphas):
            # Calculate the BDFE for the current rank
            BDFE, _ = cmn.calc_BDFE(alpha, h, J)
            bdfes[i].extend(BDFE)

    return bdfes

