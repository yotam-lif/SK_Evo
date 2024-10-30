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
    compute_alpha_from_hist,
    calc_basic_lfs
)

def main():
    # Parameters
    N_start = 800
    N_stop = 1600
    N_step = 200
    random_state = 42  # Seed for reproducibility
    beta = 1.0
    rho = 1.0  # Sparsity of the coupling matrix
    n_times = 100  # Example value, adjust as needed
    n_repeats = 5  # Number of times to repeat per each N value, for averaging
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dir_path = os.path.join(dir_path, "Plots", "lf_sq_mean")
    os.makedirs(dir_path, exist_ok=True)

    num_N = int((N_stop - N_start) / N_step)
    N_values = np.linspace(N_start, N_stop, num_N + 1, dtype=int)
    mean_lf_sq_all = np.zeros((num_N + 1, n_times))

    for n_ind, N in enumerate(N_values):
        print(f"Running for N = {N}")
        mean_lf_sq_repeats = np.zeros((n_repeats, n_times))
        fitness_repeats = np.zeros((n_repeats, n_times))
        for i in range(n_repeats):
            # Initialize the model
            alpha = init_alpha(N)
            h = init_h(N, random_state=random_state, beta=beta)
            J = init_J(N, random_state=random_state, beta=beta, rho=rho)
            F_off = calc_F_off(alpha, h, J)

            flip_seq = relax_sk(
                alpha=alpha.copy(),
                his=h,
                Jijs=J,
                sswm=True  # Change to False to use Glauber flip
            )

            num_flips = len(flip_seq)
            flips = np.linspace(0, num_flips, n_times, dtype=int)
            saved_alphas = [compute_alpha_from_hist(alpha, flip_seq, flip) for flip in flips]

            mean_lf_sq = np.zeros(n_times)
            fitness_list = np.zeros(n_times)
            for n_time in range(n_times):
                alpha_i = saved_alphas[n_time]
                lfs = Funcs.calc_basic_lfs(alpha_i, h, J)
                mean_lf_sq[n_time] = np.mean(lfs ** 2)
                fitness_list[n_time] = compute_fit_slow(alpha_i, h, J, F_off)

            mean_lf_sq_repeats[i] = mean_lf_sq
            fitness_repeats[i] = fitness_list

        mean_lf_sq_all[n_ind] = np.mean(mean_lf_sq_repeats, axis=0)
        mean_fitness_all = np.mean(fitness_repeats, axis=0)

        # Plot the log of mean_lf_sq as a function of fitness for each N
        plt.figure(figsize=(10, 8))
        plt.plot(np.log(mean_fitness_all), np.log(mean_lf_sq_all[n_ind]), 'o', label=f"N = {N}")
        plt.xlabel("log(Fitness)")
        plt.ylabel("log(Mean Local Field Squared)")
        plt.title(f"Log-Log Plot of Mean Local Field Squared vs Fitness for N = {N}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(dir_path, f"lf_sq_mean_log_log_N_{N}.png"))
        plt.close()

if __name__ == "__main__":
    main()