#!/bin/bash
#SBATCH --partition=default_free
#SBATCH --account=comet_spread
#SBATCH --mem=2GB
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00
#SBATCH --output=data/diffusion/output_diffusion.log

# Load modules

module load Python/3.12.3-GCCcore-13.3.0

# Activate virtual environment
source .venv/bin/activate

# Set temp dir
TEMPDIR=/nobackup/proj/comet_spread/

date
echo "Running on $HOSTNAME SPRE analysis"

#../new_knockoffgwas_pipeline/run_pre_knockoff_gwas.sh $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID $DATA/Nicola pbc 0.1 results 2.5 3

# Run diffusion model
python src/run_model_analysis.py data/diffusion/input_diffusion_$1.json

#echo "Node memory state: `free`"
date

echo Total seconds: $SECONDS
