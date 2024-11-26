import os
import numpy as np
import matplotlib.pyplot as plt
from misc import cmn as Fs

def plot_crossings(flips_labels, anc_bdfe, prop_anc_bdfe, evol_bdfe, prop_evol_bdfe, plot_dir, N, beta, rho):
    """
    Plot crossings between two consecutive flips.

    Parameters:
        flips_labels (list): Labels for the two flips.
        anc_bdfe (np.ndarray): Ancestral BDFE values.
        prop_anc_bdfe (np.ndarray): Proposed Ancestral BDFE values.
        evol_bdfe (np.ndarray): Evolved BDFE values.
        prop_evol_bdfe (np.ndarray): Proposed Evolved BDFE values.
        plot_dir (str): Directory to save plots.
        N (int): Number of spins (for plot title).
    """
    plt.figure(figsize=(10, 8))

    # Plot ancestral DFEs
    for j in range(len(anc_bdfe)):
        plt.plot(flips_labels, [anc_bdfe[j], prop_anc_bdfe[j]], color='coral', alpha=0.2)
    plt.scatter([flips_labels[0]] * len(anc_bdfe), anc_bdfe, color='coral', edgecolor='coral', s=20, facecolors='none', label='Forwards')
    plt.scatter([flips_labels[1]] * len(prop_anc_bdfe), prop_anc_bdfe, color='coral', edgecolor='coral', s=20, facecolors='none')

    # Plot evolved DFEs
    for j in range(len(evol_bdfe)):
        plt.plot(flips_labels, [prop_evol_bdfe[j], evol_bdfe[j]], color='royalblue', alpha=0.2)
    plt.scatter([flips_labels[0]] * len(prop_evol_bdfe), prop_evol_bdfe, color='royalblue', edgecolor='royalblue', s=20, facecolors='none', label='Backwards')
    plt.scatter([flips_labels[1]] * len(evol_bdfe), evol_bdfe, color='royalblue', edgecolor='royalblue', s=20, facecolors='none')

    # Customize plot
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Flip")
    plt.ylabel("Fitness (s)")
    plt.title(f"Crossings between {flips_labels[0]} - {flips_labels[1]}; N = {N}; β = {beta}; ρ = {rho}")
    plt.tight_layout()
    plt.legend()

    # Save plot
    plot_path = os.path.join(plot_dir, f"crossings_{flips_labels[0].replace(' ', '_')}_{flips_labels[1].replace(' ', '_')}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()  # Close the figure to free memory

def main():
    N = 3000  # Number of spins
    beta = 1.0  # Epistasis strength
    rho = 1.0  # Fraction of non-zero coupling elements
    num_stops = 5

    # Initialize spin configuration
    alpha_initial = Fs.init_alpha(N)

    # Initialize external fields
    h = Fs.init_h(N, beta=beta)

    # Initialize coupling matrix with sparsity
    J = Fs.init_J(N, beta=beta, rho=rho)

    # Create main output directory for plots
    proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plot_dir = os.path.join(proj_dir, "Plots", "scrambling")
    os.makedirs(plot_dir, exist_ok=True)

    # Relax the system using sswm_flip (sswm=True)
    flip_seq = Fs.relax_sk(alpha_initial.copy(), h, J, sswm=True)
    flip_num = len(flip_seq)
    flips = np.linspace(0, flip_num, num=num_stops, dtype=int)

    # Ensure that flip indices are unique and sorted
    flips = sorted(set(flips))
    num_stops = len(flips)

    # Compute saved alphas
    saved_alphas = [Fs.compute_alpha_from_hist(alpha_initial, flip_seq, flip) for flip in flips]

    dfes = []
    bdfes = []
    for alpha in saved_alphas:
        dfe = Fs.calc_DFE(alpha, h, J)
        dfes.append(dfe)
        bdfe, bdfe_ind = Fs.calc_BDFE(alpha, h, J)
        bdfes.append((bdfe, bdfe_ind))

    # Plot each pair of consecutive DFEs
    for i in range(num_stops - 1):
        anc = dfes[i]
        evol = dfes[i + 1]
        anc_bdfe, anc_bdfe_ind = bdfes[i]
        evol_bdfe, evol_bdfe_ind = bdfes[i + 1]

        # Extract proposed DFEs based on indices
        prop_anc_bdfe = evol[anc_bdfe_ind]
        prop_evol_bdfe = anc[evol_bdfe_ind]

        # Prepare data for plotting
        flips_labels = [f"flip {flips[i]}", f"flip {flips[i + 1]}"]

        # Plot crossings for the current rank pair
        plot_crossings(
            flips_labels,
            anc_bdfe,
            prop_anc_bdfe,
            evol_bdfe,
            prop_evol_bdfe,
            plot_dir,
            N,
            beta,
            rho
        )

    print("Plots have been saved successfully.")

if __name__ == "__main__":
    main()
