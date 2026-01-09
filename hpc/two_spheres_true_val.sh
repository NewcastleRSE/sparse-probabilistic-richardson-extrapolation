#!/bin/bash
#SBATCH --partition=long_free
#SBATCH --account=comet_spread
#SBATCH --mem=2GB
#SBATCH --cpus-per-task=1
#SBATCH --output=data/mujoco/results/output_two_spheres_true_2_val.log

# Load modules

module load Python/3.12.3-GCCcore-13.3.0

# Activate virtual environment
source .venv/bin/activate

# Set temp dir
TEMPDIR=/nobackup/proj/comet_spread/

date
echo "Running on $HOSTNAME calc. true value"

# Run physics model for true value
python src/simulate_model.py data/mujoco/input_two_spheres_true_2.json 1

#echo "Node memory state: `free`"
date

echo Total seconds: $SECONDS
