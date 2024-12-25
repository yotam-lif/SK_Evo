#!/bin/bash

#BSUB -q short                   # Specify the queue
#BSUB -m "cn1[13-26]"            # Restrict to specific compute nodes
#BSUB -o out.txt                 # Redirect standard output to out.txt
#BSUB -e err.txt                 # Redirect standard error to err.txt

# Load Miniconda module
module load miniconda/24.9.2_environmentally

# Activate the Conda environment
conda activate my_env

# Run the Python script
python misc/generate_data.py --N 2000 --beta 1.0 --rho 0.05 --n_repeats 100 --output_dir './run_data'

# Deactivate the environment after the script finishes
conda deactivate