##############################################################################
# Script to simulate model for the given parameter file with ith 
# discretisation parameters (given by X and h arrays).
# This can be called as an array batch script on HPC.
# The results will then be added to the model
# results cache which can then be used later with SPRE.
#
# From root directory, for example run
# python ./src/simulate_model.py ./data/diffusion/input_diffusion_38.json 2
#
# To simulate the model using the 2nd discretisation parameters that would
#  be simulated for the given X and h arrays.
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import sys

# Application modules
import models

# ----------------------------------------------------------
# Read in parameters and options for running SPRE with various tolerences
# ----------------------------------------------------------
# Get new filename
parameter_filename = sys.argv[1]

# Get which set of discretisation parameters to use
sim_number = int(sys.argv[2])

# Get model object
model = models.get_model(parameter_filename, skip_true_value_calc = True)

# Simulate the model and add outcome to cache
model.simulate_ith_analysis_setting(sim_number)
