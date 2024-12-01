# generate_data.py

import cmn
import os
import pickle

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

    for i in range(n_repeats):
        init_alpha = cmn.init_alpha(N)
        h = cmn.init_h(N, beta=beta)
        J = cmn.init_J(N, beta=beta, rho=rho)
        flip_seq = cmn.relax_sk(init_alpha.copy(), h, J, sswm=True)

        data.append({
            'init_alpha': init_alpha,
            'h': h,
            'J': J,
            'flip_seq': flip_seq
        })

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the data to a file
    output_file = os.path.join(output_dir, f'N{N}_rho{int(rho*100)}_beta{int(beta*100)}_repeats{n_repeats}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"Data saved to {output_file}")

def main():
    """
    Main function to generate data with predefined parameters.
    """
    N = 4000
    beta = 1.0
    rho = 1.0
    n_repeats = 50
    output_dir = 'run_data'

    generate_data(N, beta, rho, n_repeats, output_dir)

if __name__ == "__main__":
    main()