##############################################################################
# Utility functions for models
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import json

# Application modules
from models.base_model import Model
from models.models_mujoco import *
from models.models_ode import *
from models.models_pde import *
from models.models_pybullet import *
from models.models_toy import *

def get_model(parameter_filename : str, skip_true_value_calc : bool = False) -> Model:
    """
    Returns model object for the appropriate model as stated in the model parameter file.

    Parameters:  
        parameter_filename : str     Filename and path of the file.  
        skip_true_value_calc : bool  Skip evaulation of the true value (if not doing SPRE)
    Returns:
        Model
    """
    
    # Get parameters
    with open(parameter_filename) as f:
        params = json.load(f)

    # Get model name and set up model object
    if "model_name" in params.keys():
        model_name = params["model_name"]
    else:
        model_name = "sir"

    # Get model class name
    parts = model_name.split('_')
    model_class_name = ''.join(word.capitalize() for word in parts)  + "Model"

    # Create model object
    model = globals()[model_class_name](params, parameter_filename, skip_true_value_calc)

    return model
