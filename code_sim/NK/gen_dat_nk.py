import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from code_sim.cmn import cmn_nk, cmn

output_dir = '../misc/run_data/NK'
os.makedirs(output_dir, exist_ok=True)

def process(N, k):
    init_sigma = cmn.init_sigma(N)
    NK_init = cmn_nk.NK(N, k)
    flip_seq, NK = cmn_nk.relax_nk(init_sigma, NK_init)
    return {'flip_seq': flip_seq, 'NK': NK}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate NK model data.')
    parser.add_argument('N', type=int, help='Number of loci')
    parser.add_argument('k', type=int, help='Number of neighbors per locus')
    parser.add_argument('num_repeats', type=int, help='Number of repeats')
    args = parser.parse_args()

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process, args.N, args.k) for _ in range(args.num_repeats)]
        for future in futures:
            results.append(future.result())

    output_path = os.path.join(output_dir, f'N_{args.N}_K_{args.k}_repeats_{args.num_repeats}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)