import os
import numpy as np
import matplotlib.pyplot as plt

import Funcs
from Funcs import (
    init_alpha,
    init_h,
    init_J,
    relax_sk,
    compute_fit_slow,
    calc_F_off,
    compute_alpha_from_hist
)

def main():
    # Parameters
    N_start = 800
    N_stop = 1600
    N_step = 50

    random_state = 42  # Seed for reproducibility
    beta = 1.0
    rho = 1.0  # Sparsity of the coupling matrix
    # Define the number of lowest ranks to plot
    n_times = 200  # Example value, adjust as needed
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dir_path = os.path.join(dir_path, "Plots", "pair_corr_dist")
    os.makedirs(dir_path, exist_ok=True)

    num_N = int((N_stop - N_start) / N_step)
    N = np.linspace(N_start, N_stop, num_N + 1, dtype=int)

    for n in N:
        # Initialize the model
        alpha = init_alpha(n)
        h = init_h(n, random_state=random_state, beta=beta)
        J = init_J(n, random_state=random_state, beta=beta, rho=rho)
        F_off = calc_F_off(alpha, h, J)

        flip_seq = relax_sk(
            alpha=alpha.copy(),
            his=h,
            Jijs=J,
            sswm=True  # Change to False to use Glauber flip
        )

        num_flips = len(flip_seq)
        flips = np.linspace(0, num_flips, n_times, dtype=int)
        saved_alphas = [compute_alpha_from_hist(alpha, flip_seq, flips[flip]) for flip in flips]


        fitness_list = [compute_fit_slow(alpha_i, h, J, F_off) for alpha_i in saved_alphas]
        f_i_sq_mean = np.zeros(len(saved_alphas))
        for n_time in range(len(saved_alphas)):
            alpha_i = saved_alphas[n_time]
            # The pairwise terms sum per i can be rewritten as (lf_i)^2 - N*sig_J^2 = (lf_i)^2 - beta
            # Compute J[i, :] * alpha_i for all i
            lfs = Funcs.calc_basic_lfs(alpha_i, h, J)
            lfs = lfs ** 2
            f_i_sq_mean[n_time] = np.mean(lfs)

        f_i_sq_mean = np.log(f_i_sq_mean)
        # fitness_list = np.log(fitness_list)
        f_i_sq_mean =  f_i_sq_mean[int(n_times/2.5):]
        fitness_list = np.array(fitness_list[int(n_times/2.5):])

        slope, intercept = np.polyfit(fitness_list, f_i_sq_mean, 1)
        fit_line = slope * fitness_list + intercept

    plt.figure(figsize=(8, 6))
    plt.plot(fitness_list, f_i_sq_mean, label="Mean local field squared")
    plt.plot(fitness_list, fit_line, label="Fit line")
    plt.text(0.5, 0.5, f"Slope*N: {slope*N:.2f}, intercept: {intercept:.2f}", fontsize=12, transform=plt.gca().transAxes)
    plt.xlabel("Fitness")
    plt.ylabel("$\\langle f_i^2 \\rangle$")
    plt.title(f"$\\langle f_i^2 \\rangle$ vs Fitness; N: {N}")
    plt.show()


if __name__ == "__main__":
    main()