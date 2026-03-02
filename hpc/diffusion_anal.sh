#!/bin/bash
#
# Submit with: sbatch hpc/ball_array_spre <scenario_number>
#
#SBATCH --partition=long_free
#SBATCH --account=comet_spread
#SBATCH --mem=5GB
#SBATCH --cpus-per-task=1
#SBATCH --output=data/diffusion/results/output_diffusion_anal.log


# Load modules

module load Python/3.12.3-GCCcore-13.3.0

# Activate virtual environment
source .venv/bin/activate

# Set temp dir
TEMPDIR=/nobackup/proj/comet_spread/

date
echo "Running on $HOSTNAME analysis for scenario $1"


# Run model
python src/run_model_analysis.py data/diffusion/input_diffusion_$1.json

#echo "Node memory state: `free`"
date

echo Total seconds: $SECONDS
