#!/bin/bash
#
# Submit with: sbatch hpc/ball_array_spre <scenario_number>
#
#SBATCH --partition=default_free
#SBATCH --account=comet_spread
#SBATCH --mem=5GB
#SBATCH --cpus-per-task=1
#SBATCH --output=data/flock/results/output_flock_anal_%a.log

# Load modules

module load Python/3.12.3-GCCcore-13.3.0

# Activate virtual environment
source .venv/bin/activate

# Set temp dir
TEMPDIR=/nobackup/proj/comet_spread/

date
echo "Running on $HOSTNAME analysis for scenario $1"

# Make copy of parameter file
cp data/flock/input_flock_$1.json data/flock/input$2_flock_$1.json

# Change the seed
sed -i "s/\"seed\": 1/\"seed\": $2/" data/flock/input$2_flock_$1.json

# Change output filenames
sed -i "s/output/output$2/" data/flock/input$2_flock_$1.json

# Run model
python src/run_model_analysis.py data/flock/input$2_flock_$1.json

#echo "Node memory state: `free`"
date

# Delete old parameter file
rm data/flock/input$2_flock_$1.json

echo Total seconds: $SECONDS

