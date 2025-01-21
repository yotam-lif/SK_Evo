import matplotlib.pyplot as plt
from code.cmn import cmn, cmn_sk
import numpy as np
import seaborn as sns

N = 1500
random_state = 42
beta = 1.0
rho = 1.0
num_points = 10
fis = []

def main():
    sigma0 = cmn.init_sigma(N)
    h = cmn_sk.init_h(N, random_state=random_state, beta=beta)
    J = cmn_sk.init_J(N, random_state=random_state, beta=beta, rho=rho)
    flip_seq = cmn_sk.relax_sk(sigma0, h, J)
    num_flips = len(flip_seq)
    ts = np.linspace(0, num_flips, num_points, dtype=int)
    sigmas = cmn.curate_sigma_list(sigma0, flip_seq, ts)
    for sigma in sigmas:
        fi = cmn_sk.compute_fis(sigma, h, J)
        fis.append(fi)
    for i, fi in enumerate(fis):
        plt.figure()
        sns.histplot(fi, kde=False, bins=40, color='grey', alpha=0.5, stat='density', element="step", edgecolor="black")
        plt.title(f'$t={int(ts[i]/ts[-1] * 100)} \\% $')
        plt.show()

if __name__ == "__main__":
    main()