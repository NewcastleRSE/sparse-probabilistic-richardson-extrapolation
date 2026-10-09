##############################################################################
# Script to simulate models for two spheres model on a grid as defined below.
#  
# The results will then be stored in a results file.
#
# From root directory, for example run
# python ./src/simulate_models_timings_grid.py 
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import time
import numpy as np
import pandas as pd

# Application modules
from models.models_utils import get_model

# Get model parameter settings, true value is 0.4753407466699643
parameter_filename = "./data/mujoco/input_two_spheres_300.json"

# Results filename
results_filename = "./data/mujoco/results/two_spheres_300_timings_grid.dat"

# Get model object
model = get_model(parameter_filename, skip_true_value_calc = True)

# Update to not use the model cache 
model.use_model_cache = False

all_results = []

# Set min and max for each parameter

# Start of loops
# Time loop
for dt in np.linspace(1e-12, 1e-11, 10):
    for solver_reference in np.linspace(1e-12, 1e-11, 10):
        for solver_impedance in np.linspace(1e-12, 1e-11, 10):
            # Set parameters to use for simulation, "dt", "solver_reference", "solver_impedance"
            parameters_to_set = np.array((dt, solver_reference, solver_impedance))

            print(f"Running model \"{model.model_name}\" with parameters {parameters_to_set}") 

            # Simulate the model and time how long it takes
            start_time = time.perf_counter() 
                    
            y = model.run_model(parameters_to_set)

            # Record timing of this model simulation
            run_time = time.perf_counter() - start_time

            # Record results to table
            result = [
                parameters_to_set[0],
                parameters_to_set[1],
                parameters_to_set[2],
                run_time,
                y
            ]

            if len(all_results) == 0:
                all_results = [result]
            else:
                all_results.append(result)

# End of loops

# Write output file
df_timings = pd.DataFrame(
    all_results,
    columns=["dt", "solver_reference", "solver_impedance", "run_time", "y"]
)

df_timings.to_csv(results_filename, sep="\t", index=False)
