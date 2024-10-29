import os
import numpy as np
import matplotlib.pyplot as plt

import Funcs
from Funcs import (
    init_alpha,
    init_h,
    init_J,
    relax_SK,
    compute_fit_slow,
    calc_F_off
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
    n_times = 100  # Example value, adjust as needed
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dir_path = os.path.join(dir_path, "Plots", "lf_sq_mean_slope_vs_N")
    os.makedirs(dir_path, exist_ok=True)

    num_N = int((N_stop - N_start) / N_step)
    N = np.linspace(N_start, N_stop, num_N + 1, dtype=int)
    slopes = []
    intercepts = []

    for n in N:
        # Initialize the model
        alpha = init_alpha(n)
        h = init_h(n, random_state=random_state, beta=beta)
        J = init_J(n, random_state=random_state, beta=beta, rho=rho)
        flips = np.linspace(0, int(n * 0.6), n_times, dtype=int)
        F_off = calc_F_off(alpha, h, J)

        final_alpha, saved_alphas, _, _, saved_fits = relax_SK(
            alpha=alpha.copy(),
            his=h,
            Jijs=J,
            flips=flips,
            sswm=True  # Change to False to use Glauber flip
        )

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
        f_i_sq_mean =  f_i_sq_mean[int(n_times/1.5):]
        fitness_list = np.array(fitness_list[int(n_times/1.5):])

        slope, intercept = np.polyfit(fitness_list, f_i_sq_mean, 1)
        slopes.append(slope)
        intercepts.append(intercept)

    slopes = np.array(slopes) * N
    slope, intercept = np.polyfit(N, slopes, 1)
    fit_line = slope * N + intercept
    plt.figure(figsize=(8, 6))
    plt.plot(N, slopes, label="Mean local field squared")
    plt.plot(N, fit_line, label="Fit line")
    plt.text(0.5, 0.5, f"Slope: {slope:.5f}, intercept: {intercept:.2f}", fontsize=12, transform=plt.gca().transAxes)
    plt.xlabel("$N$")
    plt.ylabel("$m(N)*N$")
    plt.title(f"$ln \\left( \\langle f_i^2 \\rangle \\right)/F$ vs. $N$")
    plt.show()


if __name__ == "__main__":
    main()