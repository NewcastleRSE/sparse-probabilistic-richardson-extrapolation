##############################################################################
# Script to run SPRE analysis given a model parameter file.
#
# From root directory, for example run
# python ./src/run_model_analysis.py ./data/chem_equil/input_1.json
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import sys

# Application modules
from models.models_utils import get_model

# ----------------------------------------------------------
# Read in parameters and options for running SPRE with various tolerences
# ----------------------------------------------------------
# Get new filename
parameter_filename = sys.argv[1]

# Get model object
model = get_model(parameter_filename)

# Run the analysis
model.run_analysis()
