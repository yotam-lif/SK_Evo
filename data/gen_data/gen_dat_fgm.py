import os
import pickle
from cmn.cmn_fgm import Fisher

sig = 0.05
max_steps = 3000
m = 2 * 10 ** 3
n=32
repeats = 10 ** 3
results = []

for r in range(repeats):
    model = Fisher(n=n, sigma=sig, m=m, random_state=r)
    flips, traj, dfes = model.relax(max_steps=max_steps)
    results.append({
        'flips': flips,
        'traj': traj,
        'dfes': dfes})

# Save the results to a pickle file
output_file = f'fgm_rps{repeats}_n{n}_sig{sig}_m{m}.pkl'
output_dir = '../FGM'
output_path = os.path.join(output_dir, output_file)
os.makedirs(output_dir, exist_ok=True)
with open(output_path, 'wb') as f:
    pickle.dump(results, f)
print(f"Data saved to {output_path}")