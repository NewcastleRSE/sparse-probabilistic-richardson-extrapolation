##############################################################################
# Simulation Model with simple cubic equation
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import numpy.typing as npt

# Application modules
from models.base_model import Model

class CubicModel(Model):
    """
    Simple test model using a cubic equation.
    """

    def __init__(self, params, parameter_filename, skip_true_value_calc : bool = False):
        """
        Sets up the physics model class with model parameters.

        Parameters:  
            params : dict               Parameters for the model.
            parameter_filename : str    Filename and path of the file. 
            skip_true_value_calc : bool Skip evaulation of the true value (if not doing SPRE)     
        Returns:
            None         
        """
     
        # Use model f_z(x) = f(z+x)
        self.use_offset_model = False
      
        # Set initial model description       
        self.description = "Simple Cubic Model"

        # Call Parent’s constructor to set parameters - and overwrite any above here but not below
        super().__init__(params, parameter_filename)

        # Set initial model name    
        self.model_name = "Cubic"
 
        # Set true value
        if not skip_true_value_calc:
            self.set_true_value()
    
    def run_model_simulation(self, discrete_paras):
        """
        Use Cubic equation to return "simulation" results.

        Parameters:  
            discrete_paras : npt.NDArray     Discretisation parameters used to simulate model.           
        Returns:
            float   
        """
   
        # Set discretisation parameters
        x1 = discrete_paras[0]
        x2 = discrete_paras[1]
       
        # Output info on what is being simulated
        print(f"\tSimulating {self.description} with x1 = {x1}, and x2 = {x2}")
 
     
        # Final value, y
        y = 1 + x1 + x2*x2 + 0.01 * x1 * x1 * x1

        print(f"\tCalculated final value: {y}")
        return y

    def plot_final_model(self):
        """
        Nothing to plot for this example.

        Parameters:  
            None
        Returns:
            None                
        """
        
        pass

    def get_cache_filename(self, discrete_paras : npt.NDArray):
        """
        No need to use the cache really, but included for completion.

        Parameters:  
            discrete_paras : npt.NDArray   Discretisation parameters
        Returns:
            str    
        """
      
        # Create filename with all settings and parameters used
        filename = f"cub_"

        if self.use_offset_model:
            filename += "_".join(str(i) for i in self.final_tols) + "_"

        filename += "_".join(str(i) for i in discrete_paras) + ".bin"

        return filename
    
    def set_true_value(self):
        """
        Sets the "true_value" by running the model with small discretisation parameters
        as given in self.final_tols
        
        Parameters:  
            None
        Returns:
            None                
        """

        # True value is 1
        self.true_value = 1

        # If using the offset model, f_z(x) = f(z+x). So evaluate f_z(0) = f(z)
        if self.use_offset_model:
            self.use_offset_model = False
            self.true_value = self.run_model(self.final_tols)
            # Set back as before
            self.use_offset_model = True

        

