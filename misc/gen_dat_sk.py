import argparse
from cmn import cmn, cmn_sk
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

def generate_single_data(N, beta, rho):
    init_sigma = cmn.init_sigma(N)
    h = cmn_sk.init_h(N, beta=beta)
    J = cmn_sk.init_J(N, beta=beta, rho=rho)
    flip_seq = cmn_sk.relax_sk(init_sigma, h, J, sswm=True)
    return {
        'init_sigma': init_sigma,
        'h': h,
        'J': J,
        'flip_seq': flip_seq
    }

def generate_data(N, beta, rho, n_repeats, output_dir):
    """
    Generates data for a given number of spins, epistasis strength, and coupling elements.

    Parameters:
        N (int): Number of spins.
        beta (float): Epistasis strength.
        rho (float): Fraction of non-zero coupling elements.
        n_repeats (int): Number of repeats for data generation.
        output_dir (str): Directory to save the generated data.
    """
    data = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(generate_single_data, N, beta, rho) for _ in range(n_repeats)]
        for future in futures:
            data.append(future.result())

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the data to a file
    output_file = os.path.join(output_dir, f'N{N}_rho{int(rho*100)}_beta{int(beta*100)}_repeats{n_repeats}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate data for spins, epistasis strength, and coupling elements.')
    parser.add_argument('--N', type=int, required=True, help='Number of spins')
    parser.add_argument('--beta', type=float, required=True, help='Epistasis strength')
    parser.add_argument('--rho', type=float, required=True, help='Fraction of non-zero coupling elements')
    parser.add_argument('--n_repeats', type=int, required=True, help='Number of repeats for data generation')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the generated data')

    args = parser.parse_args()

    generate_data(args.N, args.beta, args.rho, args.n_repeats, args.output_dir)