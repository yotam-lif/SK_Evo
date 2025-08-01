import os
import pickle
from cmn.cmn_fgm import Fisher

delta = 10 ** -4
max_steps = 3000
m = 2 * 10 ** 3
sig_0 = 10 ** -3
n=4
repeats = 100
results = []

for r in range(repeats):
    model = Fisher(n=n, delta=delta, m=m, sig_0=sig_0, random_state=r, isotropic=True)
    flips, traj, dfes = model.relax(max_steps=max_steps)
    results.append({
        'flips': flips,
        'traj': traj,
        'dfes': dfes})

# Save the results to a pickle file
output_file = f'fgm_rps{repeats}_del{delta}_s0{sig_0}.pkl'
output_dir = '../FGM'
output_path = os.path.join(output_dir, output_file)
os.makedirs(output_dir, exist_ok=True)
with open(output_path, 'wb') as f:
    pickle.dump(results, f)
print(f"Data saved to {output_path}")