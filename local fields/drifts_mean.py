import os
import numpy as np
import matplotlib.pyplot as plt
from misc.cmn_sk import init_alpha, init_h, init_J, relax_sk_ranks, compute_bdfe
import scienceplots

def main():
    # -------------------------------
    # 1. Initialize SK Environment
    # -------------------------------

    # Parameters
    N_start = 1000
    N_end = 2000
    N_step = 250
    N_num = int((N_end - N_start) / N_step) + 1
    N = np.linspace(N_start, N_end, N_num, dtype=int)  # Number of spins
    beta = 1.0  # Epistasis strength
    rho = 1.0  # Fraction of non-zero coupling elements
    random_state = 42  # Seed for reproducibility
    num_saves = 30

    plt.style.use('science')

    plt.figure()
    plt.xlabel("$r(t) / N$")
    plt.ylabel(f"$ N\\langle d_{{ij}} \\rangle $ for unstable $i$")
    # plt.title("Mean Drifts vs. Rank")
    plt.gca().invert_xaxis()  # Reverse the x-axis

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dir_path = os.path.join(dir_path, "Plots", "drift_means")
    os.makedirs(dir_path, exist_ok=True)

    for n in N:
        # Initialize the model
        alpha = init_alpha(n)
        h = init_h(n, random_state=random_state, beta=beta)
        J = init_J(n, random_state=random_state, beta=beta, rho=rho)

        # Perform relaxation
        final_alpha, saved_alphas, ranks = relax_sk_ranks(
            alpha=alpha.copy(),
            his=h,
            Jijs=J,
            num_ranks=num_saves,
            sswm=True  # Change to False to use Glauber flip
        )

        # -------------------------------
        # 2. Calculate Drifts
        # -------------------------------
        # We want to look at the terms in deltas, Delta_ij.
        # We want to look at the mean of these terms for i's, such that Delta_i > 0, i.e. unstable spins.
        means = []
        valid_ranks = []
        for rank, alpha_i in zip(ranks, saved_alphas):
            if alpha_i is not None:
                Delta_ij = np.outer(alpha_i, alpha_i)
                Delta_ij = -2 * np.multiply(Delta_ij, J)
                bdfe, bdfe_ind = compute_bdfe(alpha_i, h, J)
                # Get Delta_ij with rows only corresponding to indexes in bdfe_ind
                Delta_ij = Delta_ij[bdfe_ind]
                # Get the mean of all terms
                # mean_Delta_ij = np.mean(Delta_ij, axis=1)
                mean_Delta_ij = np.sum(Delta_ij, axis=1)
                means.append(mean_Delta_ij)
                valid_ranks.append(rank)

        valid_ranks = np.array(valid_ranks) / n
        # means = [np.mean(mean) * n for mean in means]
        means = [np.sum(mean) / n for mean in means]
        print(f'N={n}, means not nan = {len(means)}')
        plt.plot(valid_ranks, means, marker='o', markersize=5, label=f"N={int(n)}")

    # -------------------------------
    # 3. Plot Drifts
    # -------------------------------
    plt.legend()
    plt.savefig(os.path.join(dir_path, "drift_sums.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    main()