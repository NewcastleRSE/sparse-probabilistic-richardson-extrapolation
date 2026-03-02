#!/bin/bash
#
# Submit with: sbatch ... <scenario_number>
#
#SBATCH --partition=long_free
#SBATCH --account=comet_spread
#SBATCH --mem=5GB
#SBATCH --cpus-per-task=1
#SBATCH --array=1-42                       # Run tasks 
#SBATCH --output=data/diffusion/results/output_diffusion_array_%a.log


# Load modules

module load Python/3.12.3-GCCcore-13.3.0

# Activate virtual environment
source .venv/bin/activate

# Set temp dir
TEMPDIR=/nobackup/proj/comet_spread/

date
echo "Running on $HOSTNAME model simulation for job $SLURM_ARRAY_TASK_ID scenario $1"


# Run model
python src/simulate_model.py data/diffusion/input_diffusion_$1.json $SLURM_ARRAY_TASK_ID

#echo "Node memory state: `free`"
date

echo Total seconds: $SECONDS
