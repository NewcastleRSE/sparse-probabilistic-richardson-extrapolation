#!/bin/bash
#SBATCH --partition=default_free
#SBATCH --account=comet_spread
#SBATCH --mem=2GB
#SBATCH --cpus-per-task=1
#SBATCH --time=60:00
#SBATCH --output=data/physics_mug/results/output_physics_mug.log

# Load modules

module load Python/3.12.3-GCCcore-13.3.0

# Activate virtual environment
source .venv/bin/activate

# Set temp dir
TEMPDIR=/nobackup/proj/comet_spread/

date
echo "Running on $HOSTNAME SPRE analysis"

# Run physics mug model
python src/run_model_analysis.py data/physics_mug/input_physics_mug_$1.json

#echo "Node memory state: `free`"
date

echo Total seconds: $SECONDS
