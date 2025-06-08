import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from code_sim.cmn import cmn
from code_sim.cmn import cmn_pspin


def generate_single_data_pspin(N: int, p: int) -> dict:
    """
    Generate a single p-spin simulation dataset.

    Parameters
    ----------
    N : int
        Number of spins.
    p : int
        Order of the p-spin interaction.

    Returns
    -------
    dict
        A dictionary containing:
          - 'init_sigma': Initial spin configuration (np.ndarray of ±1).
          - 'Jdict': Coupling dictionary mapping index-tuples to J values.
          - 'flip_seq': List of spin indices flipped during the SSWM walk.
    """
    # Initialize RNG for this repeat
    rng = np.random.default_rng()

    # Initial configuration: ±1 spins
    init_sigma = cmn.init_sigma(N)

    # Generate all p-spin couplings
    Jdict = cmn_pspin.init_J_pspin(N, p, rng=rng)

    # Perform greedy (SSWM) adaptive walk, recording flips
    sigma = init_sigma.copy()
    flip_seq = []
    while True:
        # Pick a beneficial mutation (flip) if exists
        i = cmn_pspin._sswm_pick_beneficial(sigma, Jdict, rng)
        if i is None:
            break
        flip_seq.append(i)
        sigma[i] *= -1

    return {
        'init_sigma': init_sigma,
        'Jdict': Jdict,
        'flip_seq': flip_seq
    }


def generate_data_pspin(N: int, p: int, n_repeats: int, output_dir: str) -> None:
    """
    Generate multiple p-spin datasets in parallel and save to a pickle file.

    Parameters
    ----------
    N : int
        Number of spins.
    p : int
        Order of the p-spin interaction.
    n_repeats : int
        Number of independent repeats to simulate.
    output_dir : str
        Directory where the output pickle will be saved.
    """
    data = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(generate_single_data_pspin, N, p) for _ in range(n_repeats)]
        for future in futures:
            data.append(future.result())

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct output filename
    fname = f"N{N}_p{p}_repeats{n_repeats}.pkl"
    output_file = os.path.join(output_dir, fname)

    # Save all simulation data
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate p-spin model simulation data (SSWM adaptive walks)"
    )
    parser.add_argument(
        '--N', type=int, required=True,
        help='Number of spins (system size)'
    )
    parser.add_argument(
        '--p', type=int, required=True,
        help='Order of p-spin interaction'
    )
    parser.add_argument(
        '--n_repeats', type=int, required=True,
        help='Number of independent simulations to generate'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory to save the output pickle file'
    )

    args = parser.parse_args()
    generate_data_pspin(args.N, args.p, args.n_repeats, args.output_dir)
