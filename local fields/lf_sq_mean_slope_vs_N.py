import os
import numpy as np
import matplotlib.pyplot as plt

from misc.cmn import (
    init_alpha,
    init_h,
    init_J,
    relax_sk,
    compute_fit_slow,
    calc_F_off,
    compute_alpha_from_hist,
    calc_basic_lfs
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
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dir_path = os.path.join(dir_path, "Plots", "lf_sq_mean_slope_vs_N")
    os.makedirs(dir_path, exist_ok=True)

    n_times = 100  # Example value, adjust as needed
    n_repeats = 5  # Number of times to repeat per each N value, for averaging
    num_N = int((N_stop - N_start) / N_step)
    N = np.linspace(N_start, N_stop, num_N + 1, dtype=int)
    slopes = np.zeros((n_repeats, num_N + 1))

    # Big loop, per N value
    for n_ind, n in enumerate(N):
        print(f"Running for N = {n}")
        for i in range(n_repeats):
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
            flips = np.linspace(int(num_flips/2), num_flips, n_times, dtype=int)
            saved_alphas = [compute_alpha_from_hist(alpha, flip_seq, flip) for flip in flips]
            fitness_list = [compute_fit_slow(alpha_i, h, J, F_off) for alpha_i in saved_alphas]
            lf_sq_mean = [np.mean(calc_basic_lfs(alpha_i, h, J) ** 2) for alpha_i in saved_alphas]
            lf_sq_mean = np.log(np.array(lf_sq_mean))
            slope, _ = np.polyfit(fitness_list, lf_sq_mean, 1)
            slopes[i, n_ind] = slope

    slopes = np.mean(slopes, axis=0)
    slope, intercept = np.polyfit(N, slopes, 1)
    fit_line = slope * N + intercept
    plt.figure(figsize=(8, 6))
    plt.plot(N, slopes, label="Mean local field squared")
    plt.plot(N, fit_line, label="Fit line")
    plt.text(0.5, 0.5, f"Slope: {slope:.7f}, intercept: {intercept:.2f}", fontsize=12, transform=plt.gca().transAxes)
    plt.xlabel("$N$")
    plt.ylabel("$m(N)*N$")
    plt.title(f"$N/F * ln \\left( \\langle f_i^2 \\rangle \\right)$ vs. $N$")
    plt.savefig(os.path.join(dir_path, "lf_sq_mean_slope_vs_N_cutoff.png"))


if __name__ == "__main__":
    main()