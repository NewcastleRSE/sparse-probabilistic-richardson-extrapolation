##############################################################################
# This model describes a fast chemical or biochemical equilibrium in which one substance (x(t))
#  instantaneously adjusts to a slowly changing external condition (represented by time (t)),
#  while another quantity (y(t)) is produced from (x) through a simple stoichiometric relationship. 
#
# From root directory, for example run
# python .\src\run_model_analysis.py .\data\chem_equil\input_1.json
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

# Get model object
model = models.get_model(parameter_filename)

# Run the analysis
model.run_analysis()






