import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Import the Funcs module
import Funcs as Fs


def main():
    # -------------------------------
    # 1. Initialize SK Environment
    # -------------------------------

    # Parameters
    N = 5000  # Number of spins
    beta = 1  # Epistasis strength
    rho = 1  # Fraction of non-zero coupling elements
    random_state = 42  # Seed for reproducibility

    # Initialize spin configuration
    alpha_initial = Fs.init_alpha(N)

    # Initialize external fields
    h = Fs.init_h(N, beta=beta, random_state=random_state)

    # Initialize coupling matrix with sparsity
    J = Fs.init_J(N, beta=beta, rho=rho, random_state=random_state)


    # Define desired ranks using linear spacing
    num_points = 150  # Adjust this number as needed
    max_rank = Fs.calc_rank(alpha_initial, h, J)
    min_rank = 400

    # Generate linearly spaced ranks
    ranks = np.linspace(min_rank, max_rank, num=num_points).astype(int)

    # Ensure ranks are unique and sorted in ascending order
    ranks = np.unique(ranks)

    # Relax the system using sswm_flip (sswm=False)
    # Now returns flips (number of mutations up to each rank)
    final_alpha, saved_alphas, flips = Fs.relax_SK(alpha_initial.copy(), h, J, ranks, sswm=True)

    # Calculate initial kis values
    ki_initial = Fs.calc_kis(alpha_initial, h, J)

    # Calculate initial basic local fields
    basic_local_fields_initial = Fs.calc_basic_lfs(alpha_initial, h, J)

    # Calculate absolute basic local fields
    abs_basic_local_fields_initial = np.abs(basic_local_fields_initial)

    # Debugging: Check which alphas were saved
    print("\n--- Saved Alphas Check ---")
    for rank, alpha in zip(ranks, saved_alphas):
        if alpha is not None:
            print(f"Rank {rank}: Alpha saved.")
        else:
            print(f"Rank {rank}: Alpha NOT saved.")
    print("--------------------------\n")

    # -------------------------------
    # 2. Setup Output Directories
    # -------------------------------

    # Determine the current script's directory
    curr_dir_path = os.path.dirname(os.path.realpath(__file__))

    # Define main output directory
    main_output_dir = os.path.join(curr_dir_path, "msd_plots_output")
    os.makedirs(main_output_dir, exist_ok=True)

    # Create subdirectory for MSD plots
    msd_plots_dir = os.path.join(main_output_dir, 'msd_plots')
    os.makedirs(msd_plots_dir, exist_ok=True)

    # -------------------------------
    # 3. Initialize Lists to Store Data
    # -------------------------------

    # For MSD calculations
    msd_delta_y_kis = []
    msd_delta_y_lf = []
    msd_delta_y_abs_lf = []
    mutation_counts = flips  # This serves as our "time" variable

    # -------------------------------
    # 4. Iterate Over Ranks and Compute MSDs
    # -------------------------------

    for idx, (rank, alpha) in enumerate(zip(ranks, saved_alphas)):
        if alpha is not None:
            # -------------------------------
            # 4.1 Calculate Current Values
            # -------------------------------

            # Calculate current kis values
            ki_current = Fs.calc_kis(alpha, h, J)

            # Calculate current basic local fields
            basic_local_fields_current = Fs.calc_basic_lfs(alpha, h, J)

            # Calculate current absolute basic local fields
            abs_basic_local_fields_current = np.abs(basic_local_fields_current)

            # -------------------------------
            # 4.2 Compute Displacements and MSDs
            # -------------------------------

            # For kis
            delta_y_kis = ki_current - ki_initial
            msd_kis = np.mean(delta_y_kis ** 2)
            msd_delta_y_kis.append(msd_kis)

            # For basic local fields
            delta_y_lf = basic_local_fields_current - basic_local_fields_initial
            msd_lf = np.mean(delta_y_lf ** 2)
            msd_delta_y_lf.append(msd_lf)

            # For absolute basic local fields
            delta_y_abs_lf = abs_basic_local_fields_current - abs_basic_local_fields_initial
            msd_abs_lf = np.mean(delta_y_abs_lf ** 2)
            msd_delta_y_abs_lf.append(msd_abs_lf)

    # -------------------------------
    # 5. Plot MSD vs. Time and Perform Fitting to f(x) = m * x^a
    # -------------------------------

    # Convert lists to numpy arrays
    mutation_counts = np.array(mutation_counts)
    msd_delta_y_kis = np.array(msd_delta_y_kis)
    msd_delta_y_lf = np.array(msd_delta_y_lf)
    msd_delta_y_abs_lf = np.array(msd_delta_y_abs_lf)

    # Filter out zero mutation counts to avoid division by zero or log(0)
    nonzero_indices = mutation_counts > 0
    mutation_counts_nonzero = mutation_counts[nonzero_indices]
    msd_delta_y_kis_nonzero = msd_delta_y_kis[nonzero_indices]
    msd_delta_y_lf_nonzero = msd_delta_y_lf[nonzero_indices]
    msd_delta_y_abs_lf_nonzero = msd_delta_y_abs_lf[nonzero_indices]

    # Function to perform fitting to f(x) = m * x^a
    def fit_power_law_function(x, y):
        # Define power-law function
        def power_law_func(x, m, a):
            return m * x ** a

        # Initial guess
        initial_guess = [1, 1]

        # Perform curve fitting
        try:
            popt, pcov = curve_fit(power_law_func, x, y, p0=initial_guess)
            m, a = popt

            # Calculate fitted values
            y_fit = power_law_func(x, m, a)

            # Calculate chi-squared
            # Avoid division by zero or negative y_fit
            valid_indices = y_fit > 0
            chi_squared = np.sum(((y[valid_indices] - y_fit[valid_indices]) ** 2) / y_fit[valid_indices])

            return m, a, chi_squared, y_fit
        except RuntimeError as e:
            print(f"Error during curve fitting: {e}")
            return np.nan, np.nan, np.nan, np.full_like(x, np.nan)

    # Perform fitting for MSD of kis
    m_kis, a_kis, chi_squared_kis, y_fit_kis = fit_power_law_function(mutation_counts_nonzero, msd_delta_y_kis_nonzero)

    # Perform fitting for MSD of basic local fields
    m_lf, a_lf, chi_squared_lf, y_fit_lf = fit_power_law_function(mutation_counts_nonzero, msd_delta_y_lf_nonzero)

    # Perform fitting for MSD of absolute basic local fields
    m_abs_lf, a_abs_lf, chi_squared_abs_lf, y_fit_abs_lf = fit_power_law_function(
        mutation_counts_nonzero, msd_delta_y_abs_lf_nonzero)

    # Plot MSD vs. mutations for kis
    plt.figure(figsize=(8, 6))
    plt.scatter(mutation_counts_nonzero, msd_delta_y_kis_nonzero, label='MSD (kis)', color='blue')
    plt.plot(mutation_counts_nonzero, y_fit_kis,
             label=f'm = {m_kis:.4f}, a = {a_kis:.4f}, χ² = {chi_squared_kis:.4f}', color='red')
    plt.title('MSD of Δy (kis) vs. Number of Mutations')
    plt.xlabel('Number of Mutations')
    plt.ylabel('MSD of Δy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    msd_kis_plot = 'msd_delta_y_kis_vs_mutations.png'
    plt.savefig(os.path.join(msd_plots_dir, msd_kis_plot))
    plt.close()

    # Print the fit parameters
    print(f'MSD of Δy (kis) fit: m = {m_kis:.4f}, a = {a_kis:.4f}')
    print(f'Chi-squared for MSD (kis) fit: {chi_squared_kis:.4f}')

    # Plot MSD vs. mutations for basic local fields
    plt.figure(figsize=(8, 6))
    plt.scatter(mutation_counts_nonzero, msd_delta_y_lf_nonzero, label='MSD (Basic LF)', color='green')
    plt.plot(mutation_counts_nonzero, y_fit_lf,
             label=f'm = {m_lf:.4f}, a = {a_lf:.4f}, χ² = {chi_squared_lf:.4f}', color='red')
    plt.title('MSD of Δy (Basic Local Fields) vs. Number of Mutations')
    plt.xlabel('Number of Mutations')
    plt.ylabel('MSD of Δy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    msd_lf_plot = 'msd_delta_y_lf_vs_mutations.png'
    plt.savefig(os.path.join(msd_plots_dir, msd_lf_plot))
    plt.close()

    # Print the fit parameters
    print(f'MSD of Δy (Basic LF) fit: m = {m_lf:.4f}, a = {a_lf:.4f}')
    print(f'Chi-squared for MSD (Basic LF) fit: {chi_squared_lf:.4f}')

    # Plot MSD vs. mutations for absolute basic local fields
    plt.figure(figsize=(8, 6))
    plt.scatter(mutation_counts_nonzero, msd_delta_y_abs_lf_nonzero, label='MSD (Abs Basic LF)', color='purple')
    plt.plot(mutation_counts_nonzero, y_fit_abs_lf,
             label=f'm = {m_abs_lf:.4f}, a = {a_abs_lf:.4f}, χ² = {chi_squared_abs_lf:.4f}', color='red')
    plt.title('MSD of Δy (Absolute Basic Local Fields) vs. Number of Mutations')
    plt.xlabel('Number of Mutations')
    plt.ylabel('MSD of Δy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    msd_abs_lf_plot = 'msd_delta_y_abs_lf_vs_mutations.png'
    plt.savefig(os.path.join(msd_plots_dir, msd_abs_lf_plot))
    plt.close()

    # Print the fit parameters
    print(f'MSD of Δy (Abs Basic LF) fit: m = {m_abs_lf:.4f}, a = {a_abs_lf:.4f}')
    print(f'Chi-squared for MSD (Abs Basic LF) fit: {chi_squared_abs_lf:.4f}')

    # -------------------------------
    # 6. Generate Summary Plot for MSD vs. Time (Regular Axes)
    # -------------------------------

    plt.figure(figsize=(10, 6))
    plt.plot(mutation_counts, msd_delta_y_kis, marker='o', label='kis')
    plt.plot(mutation_counts, msd_delta_y_lf, marker='s', label='Basic Local Fields')
    plt.plot(mutation_counts, msd_delta_y_abs_lf, marker='^', label='Absolute Basic Local Fields')
    plt.title('MSD of Δy vs. Number of Mutations')
    plt.xlabel('Number of Mutations')
    plt.ylabel('MSD of Δy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    msd_summary_plot = 'msd_delta_y_vs_mutations_summary.png'
    plt.savefig(os.path.join(msd_plots_dir, msd_summary_plot))
    plt.close()

    print(f"MSD vs. Number of Mutations summary plot saved as {msd_summary_plot}")


if __name__ == "__main__":
    main()