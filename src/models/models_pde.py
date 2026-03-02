##############################################################################
# Simulation Models using PDEs
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import numpy as np
import numpy.typing as npt
from pde import CartesianGrid, DiffusionPDE, ScalarField, PlotTracker

# Application modules
from models.base_model import Model

class DiffusionModel(Model):
    """
    Class for Diffusion model on a Cartesian grid.
    """

    def __init__(self, params, parameter_filename, skip_true_value_calc : bool = False):
        """
        Sets up the diffusion model class with model parameters.

        Parameters:  
            params : dict               Parameters for the model.
            parameter_filename : str    Filename and path of the file.
            skip_true_value_calc : bool Skip evaulation of the true value (if not doing SPRE)      
        Returns:
            None         
        """
         
        # End position to evaluate diffusion model
        self.end_pos = [0, 0]

        # Call Parent’s constructor to set parameters
        super().__init__(params, parameter_filename)

        # Set initial Diffussion model
        self.model_name = "Diffusion"

        # Set true value
        if not skip_true_value_calc:
            self.set_true_value()
       
    def run_model_simulation(self, discrete_paras):
        """
        Runs model simulation of diffusion model by using py-pde package and following example.
        https://py-pde.readthedocs.io/en/latest/examples_gallery/simple_pdes/cartesian_grid.html#sphx-glr-examples-gallery-simple-pdes-cartesian-grid-py
        Converts x and y discretisation parameters to the number of x and y divisions on the grid.

        Parameters:  
            discrete_paras : npt.NDArray     Discretisation parameters used to simulate model.           
        Returns:
            float   
        """

        dt = discrete_paras[0]
        num_x_partitions = int(np.round(abs(self.x_range[1] - self.x_range[0])/discrete_paras[1]))
        num_y_partitions = int(np.round(abs(self.y_range[1] - self.y_range[0])/discrete_paras[2]))
        
        # Output info on what is being simulated
        print(f"\tSimulating Diffusion Model with dt = {dt}, {num_x_partitions} x partitions and {num_y_partitions} y partitions")

        # Span of x and y, number of divisions in each dimension
        grid = CartesianGrid([self.x_range, self.y_range], [num_x_partitions, num_y_partitions])  # generate grid
        state = ScalarField(grid)  # generate initial condition
        state.insert(self.start_pos, self.start_amount)

        eq = DiffusionPDE(self.diffusivity)  # define the pde
        self.result = eq.solve(state, t_range=[0, self.total_time], dt=dt, tracker=None)
        
        y = self.get_final_quantity(discrete_paras)
        print(f"\tCalculated final value: {y}")
        return y

    def plot_final_model(self):
        """
        Plots the final simulated grid for the diffusion model.

        Parameters:  
            None
        Returns:
            None                
        """

        # Get model output to plot
        _ = self.run_model(self.final_tols)

        # Use result from solver to plot result
        plot_ref = self.result.plot(cmap="magma")
      
        # Save file
        if self.final_model_plot_filename: 
            # get the Matplotlib figure
            # Safely get the figure (works across versions)
            if hasattr(plot_ref, "get_figure"):
                fig = plot_ref.get_figure()
            elif hasattr(plot_ref, "ax"):
                fig = plot_ref.ax.figure
            elif hasattr(plot_ref, "axes"):
                fig = plot_ref.axes[0].figure
            else:
                raise AttributeError("Could not find figure in PlotReference")
  
            fig.savefig(self.final_model_plot_filename)

        if self.final_mp4_filename:
            self.record_mp4()

    def record_mp4(self):
        """
        Creates mp4 video of the simulated diffusion model.
        Requires ffmpeg program.

        Parameters:  
            None
        Returns:
            None                
        """

        # Save animataion
        dt = self.final_tols[0]
        num_x_partitions = self.final_tols[1] 
        num_y_partitions = self.final_tols[2] 
        
        # Span of x and y, number of divisions in each dimension
        grid = CartesianGrid([self.x_range, self.y_range], [num_x_partitions, num_y_partitions])  # generate grid
        state = ScalarField(grid)  # generate initial condition
        state.insert(self.start_pos, self.start_amount)

        eq = DiffusionPDE(self.diffusivity)  # define the pde

        # Save the animation
        tracker = PlotTracker(
            title="Diffusion",
            interrupts=0.01,
            movie=self.final_mp4_filename,      # specify the movie filename here
            plot_args={"cmap": "magma", "vmin": 0, "vmax": 1},        # optional additional plot args
            show=False,
            max_fps=20
        )

        # seconds = num_frames / fps = (total_time / interupts) / fps = (100 / 1) / 20 = 100 / 20 = 5 
        eq.solve(state, t_range=[0, self.total_time], dt=dt, tracker=tracker)
        
    def get_cache_filename(self, discrete_paras : npt.NDArray):
        """
        Returns the model cache filename for the diffusion based on the model parameters.

        Parameters:  
            discrete_paras : npt.NDArray   Discretisation parameters
        Returns:
            str    
        """
      
        # Create filename with all settings and parameters used
        filename = f"d_{self.diffusivity}_{self.x_range[0]}_{self.x_range[1]}_{self.y_range[0]}_{self.y_range[1]}_{self.total_time}"
        filename += f"_{self.start_pos[0]}_{self.start_pos[1]}_{self.start_amount}_"
        filename += f"_{self.end_pos[0]}_{self.end_pos[1]}_{self.start_amount}_"
        filename += "_".join(str(i) for i in discrete_paras) + ".bin"

        return filename
    
    #def set_true_value(self):
        """
        Sets the "true_value" of the diffusion model using analytic solution.
        
        Parameters:  
            None
        Returns:
            None                
        """

        # Get value at (0, 0) using analytic solution.
    #    self.true_value = self.diffusion_solution_2d(0, 0, self.total_time) 

    def get_final_quantity(self, discrete_paras : npt.NDArray) -> float:
        """
        Returns the final evaluation of the diffusion model given by value at (0, 0).

        Parameters:  
            discrete_paras : npt.NDArray        Discretisation parameters
        Returns:
            float                          
        """

        # Return value at (0,0) from simulated model.      
        #return self.result.interpolate([0, 0])
        return self.result.interpolate(self.end_pos)    
    
    def diffusion_solution_2d(self, x : float, y : float, t : float) -> float:
        """
        Analytic solution of the 2D diffusion equation for a point-source initial condition.

        Parameters
        ----------
        x : float      Spatial coordinates.
        y : float            
        t : float      Time at which to evaluate the solution (must be > 0).
      
        Returns
        -------
        c : float       Concentration value at position (x, y) and time t.
        """

        x = np.asarray(x)
        y = np.asarray(y)
        if t <= 0:
            raise ValueError("Time t must be positive for the analytic solution.")
        
        r2 = (x - self.start_pos[0])**2 + (y - self.start_pos[1])**2
        prefactor = 1.0 / (4 * np.pi * self.diffusivity * t)
        exponent = -r2 / (4 * self.diffusivity * t)
        return prefactor * np.exp(exponent)

