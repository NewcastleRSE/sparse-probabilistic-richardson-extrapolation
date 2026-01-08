##############################################################################
# Simulation Models
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import numpy as np
import numpy.typing as npt
import json
import os
import struct
from pathlib import Path
import pandas as pd
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pde import CartesianGrid, DiffusionPDE, ScalarField, PlotTracker
import pybullet
import pybullet_data
import mujoco
import imageio
import time

# Application modules
from sparse_pre.extrapolation import extrapolation

class Model:
    """
    Base model class with common methods used for all model classes.
    """

    def __init__(self, params : dict, parameter_filename : str, skip_true_value_calc : bool = False):
        """
        Sets up the model class with model parameters.

        Parameters:  
            params : dict               Parameters for the model.
            parameter_filename : str    Filename and path of the file.
            skip_true_value_calc : bool Skip evaulation of the true value (if not doing SPRE)
                                        Used in sub classes of Model    
        Returns:
            None         
        """
         
        # Set default parameters values  
        self.total_time = 120
        self.evaluation = False
        self.model_name = "Model not set"
        self.use_model_cache = True
        self.use_fixed_basis = False

        # Set filenames to blank by default
        self.results_plot_filename = ""
        self.results_filename = ""
        self.results_plot_filename = ""
        self.final_model_plot_filename = ""
        self.final_mp4_filename = ""
        self.results_fx_filename = ""

        # Model labels
        self.xlabel = 'time'
        self.ylabel = 'y'
        self.title = 'Model'

        # Set parmaters
        self.set_parameters(params)

        # Update paths for result files and plots
        self.update_paths(parameter_filename)

    def __str__(self) -> str:
        """
        Returns string to describe model.

        Parameters:  
            None  
        Returns:
            None         
        """

        return self.model_name
    
    def diff_model(self, t : float, y : tuple) -> tuple:
        """
        Returns current gradients of model variables. Used for differential equation models.

        Parameters:  
            t : float            Current time
            y : tuple            Current values model variables
        Returns:
            tuple                Current gradients
        """

        return (0, 0)
       
    def get_initial_condition(self) -> tuple:
        """
        Returns initial condition of a model - often set with model parameters.

        Parameters:  
            None
        Returns:
            tuple                Initial values of model variables.
        """

        return (0, 0)
    
    def get_final_quantity(self, discrete_paras : npt.NDArray) -> float:
        """
        Returns the final evaluation of a model, derived from the final state of the model.

        Parameters:  
            discrete_paras : npt.NDArray        Discretisation parameters
        Returns:
            float   
        """

        return 0
    
    def set_parameters(self, parameters : dict):
        """
        Set model parameters from a dictionary.
        Some unset parameters are then given default values.

        Parameters:  
            parameters : dict        Dictionary of parameter values
        Returns:
            None                
        """
         
        for key, value in parameters.items():
            setattr(self, key, value)

        # Check if running evalution of SPRE method, calculate absolute errors with true value.
        if "evaluation" not in parameters.keys():
            self.evaluation = False

        if "final_model_plot_filename" not in parameters.keys():
            self.final_model_plot_filename = None
        
        if "final_mp4_filename" not in parameters.keys():
            self.final_mp4_filename = None

        if "results_fx_filename" not in parameters.keys():
            self.results_fx_filename = None

    # Files to save results
    def add_path(self, path : str, filename : str):
        """
        Sets up a file with path to save results or a plot to.
        Returns an empty string if filename is not set.

        Parameters:  
            path : str        Path of where to store results.
            filename : str    Filename of results.
        Returns:
            None                
        """

        new_filename = ""
        if filename is not None and filename:
            new_filename = os.path.join(path, filename)
        return new_filename
        
    def update_paths(self, parameter_filename : str):
        """
        Updates the paths of all filenames where a result/plot is stored.

        Parameters:  
            parameter_filename : str    Filename and path of the model parameter file.  
        Returns:
            None                
        """

        # Get the directory of the input file
        input_dir = Path(parameter_filename)
  
        # Add cache directory if it does not exist
        self.cache_dir = input_dir.parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Add results directory if it does not exist
        results_dir = input_dir.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        self.results_filename = self.add_path(results_dir, self.results_filename)
        self.results_plot_filename = self.add_path(results_dir, self.results_plot_filename)
        self.final_model_plot_filename = self.add_path(results_dir, self.final_model_plot_filename)
        self.final_mp4_filename = self.add_path(results_dir, self.final_mp4_filename)
        self.results_fx_filename = self.add_path(results_dir, self.results_fx_filename)

        self.do_results_plot = self.results_plot_filename != ""
        self.do_final_model_plot = self.final_model_plot_filename != ""

        if self.evaluation:
            self.results_eval_filename = self.add_path(results_dir, self.results_eval_filename)
            self.results_eval_plot_filename = self.add_path(results_dir, self.results_eval_plot_filename)
            self.do_results_eval_plot = self.results_eval_plot_filename != ""

    def set_true_value(self):
        """
        Sets the object variable "true_value" to the actual model outcome.
        For example, from an analytic solution if known. This can be used to evaluate model accuracy.

        Parameters:  
            None
        Returns:
            None                
        """

        self.true_value = 0

    def get_cache_filename(self, discrete_paras : npt.NDArray) -> str:
        """
        Returns the model cache filename based on the model parameters.

        Parameters:  
            discrete_paras : npt.NDArray   Discretisation parameters
        Returns:
            str    
        """

        return self.model_name + "_".join(str(i) for i in discrete_paras) + ".bin"

    def update_model_cache(self, discrete_paras : npt.NDArray, y : float):
        """
        Updates the model cache of final outcome values.

        Parameters:  
            discrete_paras : npt.NDArray     Discretisation parameters used to simulate model.
            y : float                        Final calculated value.
        Returns:
            None   
        """

        # Get the cache filename and then add the path.
        cache_filename = self.get_cache_filename(discrete_paras)
        cache_filename = self.add_path(self.cache_dir, cache_filename)

        # Write the value to file in binary to store the precise number.
        with open(cache_filename, "wb") as f:
            f.write(struct.pack('<d', y))   # explicit byte order, 'd' = double (64-bit float)
       
    def run_model(self, discrete_paras : npt.NDArray) -> float:
        """
        Either runs the model or looks up previously simulated value in the cache.
       
        Parameters:  
            discrete_paras : npt.NDArray     Discretisation parameters used to simulate model.           
        Returns:
            float   
        """
   
        # If using offset model f_z(x) = f(z+x)
        if self.use_offset_model:
            discrete_paras = discrete_paras + self.final_tols

        # Whether to finally simulate the model or not.
        perform_model_simulation = True

        # Get the cache filename and then add the path.
        cache_filename = self.get_cache_filename(discrete_paras)
        cache_filename = self.add_path(self.cache_dir, cache_filename)

        # Use the cache unless requested not to.
        if self.use_model_cache:
            
            # Look up the value in the cache if it exists
            if os.path.exists(cache_filename):
                with open(cache_filename, "rb") as f:
                    data = f.read()
                                       
                    if len(data) != 8:
                        raise ValueError("Corrupted cache file! Recalculating...")
                    else:
                        y_result = struct.unpack('<d', data)[0]     
                        perform_model_simulation = False
                        print(f"\tUsing cached value: {y_result}")         

        # Value was not in cache or req'd to simulate again.     
        if perform_model_simulation:
            # Run the model
            y_result = self.run_model_simulation(discrete_paras)
            # Record result in the cache
            self.update_model_cache(discrete_paras, y_result)

        return y_result

    def run_model_simulation(self, discrete_paras : npt.NDArray) -> float:
        """
        Runs model simulation by solving differiental equations.
        Other models may override this to simulate models otherwise.
        The final outcome of the model is returned.
       
        Parameters:  
            discrete_paras : npt.NDArray     Discretisation parameters used to simulate model.           
        Returns:
            float   
        """

        y0 = self.get_initial_condition()
      
        # Time span to evalute the model
        t_span = (0, self.total_time)

        # Timepoints at which to store values
        number_of_points = 100000 #max(2, int(1/diff_tol)) + 1
        t_eval = np.linspace(*t_span, number_of_points)

        # -----------------------------
        # Solve system together
        # -----------------------------
        sol = solve_ivp(
            fun=self.diff_model,
            t_span=t_span,
            y0=y0,           
            t_eval=t_eval,
            method='RK45',
            rtol=discrete_paras[0]
            #method='LSODA',
            #min_step=discrete_paras[0],
            #max_step=discrete_paras[0],
            #initial_step=discrete_paras[0]
        )

        self.diff_solution = sol

        return self.get_final_quantity(discrete_paras)
     
    def run_analysis(self):
        """
        Runs analysis of the model by running SPRE on every set of discretisation parameters in X
        scaled for by each value in h.
        If self.evaluation is set to true evaluation results/plot is recorded comparing to the "true value".

        Parameters:  
            None
        Returns:
            None                
        """

        # Values to try
        X = np.array(self.X)

        # Apply SPRE
        # Define options
        # "name"   : str, one of {"MRE", "GRE", "SPRE"} (default: "SPRE")
        # "k_name" : str, one of {"Gaussian", "GaussianARD", "Matern1/2", "Matern3/2", "white"} (default: "white")
        options = {
            "name": self.extrapolation_name,
            "k_name":  self.extrapolation_kernel, 
            "plot" : False,
            "use_fixed_basis": self.use_fixed_basis
        }

        offset_name = ""
        if self.use_offset_model:
            offset_name = f"using offset {self.final_tols} "

        # Loop thro' different scalar values for multiplying set of tolerences
        for i, h in enumerate(self.h_values):
            # Results, Y is model output
            Y = np.array([])
            if isinstance(h, (list, tuple)):
                extrapolation_results = h.copy()
            else:
                extrapolation_results = [h]

            # Get results
            for x in X:     
                discrete_parameters = np.array(h) * np.array(x)
                print(f"Running model \"{self.model_name}\" {offset_name}with parameters {discrete_parameters}")                                    
                y = self.run_model(discrete_parameters)
                Y = np.append(Y, y)

            # Assume extrapolation is a defined function returning a dict with 'mu' and 'var'
            if self.do_results_plot: 
                filepath = Path(self.results_plot_filename.replace(".png", f"_LOOCV_{i}.png"))
                # Add LOOCV directory if it does not exist
                loocv_filepath = filepath.parent / "loocv_plots"
                loocv_filepath.mkdir(parents=True, exist_ok=True)
                # Save result in LOOCV directory
                new_filepath = filepath.parent / "loocv_plots" / filepath.name              
                options["plot_filename"] = new_filepath
            
            out = extrapolation(X*h, Y, options)

            if self.extrapolation_name != "MRE":
                print(f"Predict f(0) = {out['mu'][0]} +/- {np.sqrt(out['var'][0][0])}\n")
                extrapolation_results.extend([out['mu'][0], out['var'][0][0]])
                # Append results for each point
                extrapolation_results.extend(out['mu_cv'])
                extrapolation_results.extend(out['var_cv'])
            else:
                # No variances or cross validation to record for MRE
                print(f"Predict f(0) = {out['mu'][0]}\n")
                extrapolation_results.extend([out['mu'][0]])                      
            
            # Create table of SPRE results
            if i == 0:
                all_extrapolation_results = np.matrix(extrapolation_results)
            else:
                all_extrapolation_results = np.vstack((all_extrapolation_results, extrapolation_results))

            # Create table of absoluate errors
            if self.evaluation:
                # Create row of results for absolute error table
                # h, true_value, first_estimate, spre_estimate, abs_err_first_estimate, abs_err_spre_estimate
                if isinstance(h, (list, tuple)):
                    table_row = h.copy()
                else:
                    table_row = [h]

                table_row.extend(np.array([self.true_value, Y[0], out['mu'][0], np.abs(self.true_value - Y[0]), np.abs(self.true_value - out['mu'][0])]))

                if i == 0:
                    self.abs_error_table = np.matrix(table_row)
                else:
                    self.abs_error_table = np.vstack((self.abs_error_table, table_row))

            # Save X and Y values if filename given
            if self.results_fx_filename:
                filename = self.results_fx_filename.replace(".", f"_{i}.")               
                dataXY = np.column_stack((X*h, Y))
                header = [f"X{i+1}" for i in range(len(discrete_parameters))]
                header.extend("Y")
                header = "\t".join(header)
                # Save file
                np.savetxt(filename, dataXY, delimiter="\t", fmt="%.17g", header=header, comments='')

            # Create dataframe of results
            number_of_x = X.shape[0]

            if not isinstance(h, (list, tuple)):
                header = ["h"]
            else:
                header = [f"h{i+1}" for i in range(len(h))]

            if self.extrapolation_name != "MRE":
                header += ["mu", "var"] + [f"mu_cv{n}" for n in range(1, number_of_x + 1)] + [f"var_cv{n}" for n in range(1, number_of_x + 1)]
            else:
                header += ["mu"]

            # Create DataFrame
            self.df_all_extrapolation_results = pd.DataFrame(all_extrapolation_results, columns=header)

            # Write results to file
            if self.results_filename:
                # Write to file with tab separation
                self.df_all_extrapolation_results.to_csv(self.results_filename, sep="\t", index=False)

        
        # Do plots for the analysis if requested
        if self.evaluation:
           self.plot_evaluation_results() 

        # Do plot of SPRE estimates with error bars.
        if self.do_results_plot:
            self.plot_SPRE_results()

        # Do plot of model simulation, e.g. solved differential equations  
        if self.do_final_model_plot:
            _ = self.plot_final_model()

    def simulate_ith_analysis_setting(self, sim_number : int):
        """
        Simulates model for the set up parameters for the ith 
        discretisation parameters given by X and h arrays. The simulation outcome will then
        be added to the cache which can be used in subsequent analyses.
   
        Parameters:  
            sim_number : int
        Returns:
            None                
        """

        # Ensure the cache is not used 
        self.use_model_cache = False

        # Values to try
        X = np.array(self.X)

        # Discretisastion parameters setting count
        count = 0

        # Loop thro' different scalar values for multiplying set of tolerences
        for h in self.h_values:
           
            # Loop thro' parameters in X
            for x in X:
                count += 1
                # Only simulate this model
                if count == sim_number:     
                    discrete_parameters = np.array(h) * np.array(x)   
                    print(f"Running model \"{self.model_name}\" with parameters {discrete_parameters}")                                    
                    self.run_model(discrete_parameters)

        # Do plot of model simulation, e.g. solved differential equations  
        if self.do_final_model_plot:
            _ = self.plot_final_model()
     
       
    def choose_h_column(self, df: pd.DataFrame) -> str:
        """
        Return 'h' if present in the DataFrame.
        Otherwise return the h-like column ('h1', 'h2', ...)
        with the largest numerical range.

        Parameters:  
            df : pd.DataFrame    Dataframe of results
        Returns:
            str 
        """

        # Case 1: exact "h" exists
        if "h" in df.columns:
            return "h"

        # Case 2: search for h1, h2, h3, ... columns
        h_candidates = [col for col in df.columns
                        if col.startswith("h") and col != "h"]

        if not h_candidates:
            raise ValueError("No 'h', 'h1', 'h2', ... columns found.")

        # Helper: compute range of a column
        def col_range(col):
            return df[col].max() - df[col].min()

        # Pick column with largest spread
        best = max(h_candidates, key=col_range)

        return best

    def plot_SPRE_results(self):
        """
        Plots SPRE estimates with error bars of 1 standard deviation against different values of h.

        Parameters:  
            None
        Returns:
            None                
        """

        h_col = self.choose_h_column(self.df_all_extrapolation_results)

        # Plot with error bars
        plt.close('all') 
        plt.figure()
        # Calculate error bars (standard deviation = sqrt(var))
        if "var" in self.df_all_extrapolation_results:            
            errors = np.sqrt(self.df_all_extrapolation_results["var"])
            plt.errorbar(self.df_all_extrapolation_results[h_col], self.df_all_extrapolation_results["mu"], yerr=errors, fmt='o-', capsize=5, ecolor='black', markersize=6)
        else:
            plt.plot(self.df_all_extrapolation_results[h_col], self.df_all_extrapolation_results["mu"], 'o-', markersize=6)

        plt.xlabel("h")
        plt.ylabel("mu")
        plt.xscale('log')
        #plt.yscale('log')
        plt.title("Extrapolation Results")
        plt.grid(True)

        # Add horizontal dashed line with computed accurate answer
        if self.true_value is not None: 
            
            print(f"\nTrue value calculated as f(0) = {self.true_value}\n")

            plt.axhline(y = self.true_value, color='red', linestyle='--', linewidth=1)
    
        plt.savefig(self.results_plot_filename) 
        plt.show()    

    def plot_final_model(self):
        """
        Plots final simulated model.

        Parameters:  
            None
        Returns:
            None                
        """

        # Plot Results
        plt.close('all') 
        plt.figure(figsize=(10, 6))
        for i, y in enumerate(self.diff_solution.y):
            variable_label = self.solution_labels[i] if len(self.solution_labels[i]) > i else ''
            plt.plot(self.diff_solution.t, y, label=variable_label)
           
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if self.final_model_plot_filename:
            plt.savefig(self.final_model_plot_filename)  
        plt.show()

    def plot_evaluation_results(self):
        """
        Plots absolute errors of SPRE estimate with the "true value" as a line plot on a log-log scale plot.
        First point in X is also plotted as a reference, often with the smallest values in X, called "best estimate",
        although may not be the best estimate in X.

        Table of absolute errors is also recorded.

        Parameters:  
            None
        Returns:
            None                
        """

        # Create table of absolute errors (wrt to "true" value) with
        # 1) First point in X (smallest discretisation parameters).
        # 2) SPRE estimate using all time steps in set

        # Create DataFrame
        if not isinstance(self.h_values[0], (list, tuple)):
            abs_header = ["h"]
        else:
            abs_header = [f"h{i+1}" for i in range(len(self.h_values[0]))]
            
        abs_header += ["true_value", "first_estimate", "spre_estimate", "abs_err_first_estimate", "abs_err_spre_estimate"]
        df_abs = pd.DataFrame(self.abs_error_table, columns=abs_header)

        # Get x coordinate values to plot against
        h_col = self.choose_h_column(df_abs)
        x_vals = df_abs[h_col]
        x_lab = "h"
        if hasattr(self, "eval_plot_type"):
            if self.eval_plot_type == 2:
                x_vals = 2.0/df_abs[abs_header[2]]  
                x_lab = "grid spacing"          

        # Write results to file
        if self.results_eval_filename:
            # Write to file with tab separation
            df_abs.to_csv(self.results_eval_filename, sep="\t", index=False)

        if self.do_results_eval_plot:
            # Plot absolute errors of two estimates

            plt.close('all') 
            plt.figure()
            plt.plot(x_vals, df_abs["abs_err_first_estimate"], marker='o', linestyle='solid', linewidth=2, markersize=12, label="first estimate")
            plt.plot(x_vals, df_abs["abs_err_spre_estimate"], marker='o', linestyle='solid', linewidth=2, markersize=12, label=f"{self.extrapolation_name} estimate")
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(x_lab)
            plt.ylabel("absolute error")
            plt.title("Absolute Errors of f(0) Estimates")
            plt.grid(True)
            plt.legend()
            plt.savefig(self.results_eval_plot_filename) 
            plt.show()

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

class SirModel(Model):
    """
    Class for SIR ("Susceptible", "Infected", "Recovered") model
    """

    def __init__(self, params : dict, parameter_filename : str, skip_true_value_calc : bool = False):
        """
        Sets up the SIR model class with model parameters.

        Parameters:  
            params : dict               Parameters for the model.
            parameter_filename : str    Filename and path of the file.      
            skip_true_value_calc : bool Skip evaulation of the true value (if not doing SPRE)
        Returns:
            None         
        """

        # Set initial SIR model
        self.model_name = "SIR"
        # Model labels
        self.xlabel = 'time (days)'
        self.ylabel = 'population'
        self.title = 'SIR Model'
        self.solution_labels = ["Susceptible", "Infected", "Recovered"]

        # Call Parent’s constructor to set parameters
        super().__init__(params, parameter_filename)

        # Set true value
        if not skip_true_value_calc:
            self.set_true_value()
        
    def diff_model(self, t : float, y : tuple) -> tuple:
        """
        SIR model differential equations. Returns current gradients of S, I and R

        Parameters:  
            t : float            Current time
            y : tuple            Current values of S, I and R stored as a tuple
            beta : float         Infection rate parameter
            gamma : float        Recovery rate parameter
            N : int              Total population size
        Returns:
            tuple                Current gradients of S, I and R
        """

        S, I, R = y
        dSdt = -self.beta * S * I / self.N
        dIdt = self.beta * S * I / self.N - self.gamma * I
        dRdt = self.gamma * I
        return (dSdt, dIdt, dRdt)

    def get_initial_condition(self) -> tuple:
        """
        Returns initial condition of the SIR model.

        Parameters:  
            None
        Returns:
            tuple                Initial values of model variables.
        """
         
        S0 = self.N - self.initial_infected
        I0 = self.initial_infected
        R0 = 0
        return (S0, I0, R0)

    def set_true_value(self):
        """
        Sets the "true_value" of the SIR model by running the model with small discretisation parameters
        as given in self.final_tols
        
        Parameters:  
            None
        Returns:
            None                
        """

        self.true_value = self.run_model(self.final_tols)

    def get_final_quantity(self, discrete_paras : npt.NDArray) -> float:
        """
        Returns the final evaluation of the SIR model, derived from the final state of the model.

        Parameters:  
            discrete_paras : npt.NDArray        Discretisation parameters
        Returns:
            float                          
        """

        S, I, R = self.diff_solution.y
        t = self.diff_solution.t

        if self.evaluation:
            # Return number of recovered on last day
            return R[-1]
        else:
            # -----------------------------
            # Tolerance-controlled integral of I(t)
            # -----------------------------
            # Interpolate I(t) for smooth integration
            I_interp = interp1d(t, I, kind='cubic', fill_value="extrapolate")

            # Time span to evalute the model
            t_span = (0, self.total_time)

            # Integrate I(t) using adaptive quadrature (quad)
            infected_person_days, err = quad(I_interp, t_span[0], t_span[1], epsrel=discrete_paras[1])

            print(f"Estimate using R(120)/gamma is {R[-1]/self.gamma}\n")

            # Return the total number of "person-days of infection"
            # Indicates the burden on a helathcare system
            return infected_person_days

class ChemEquilModel(Model):
    """
    Class for Fast Chemical Equilibrium differential equation model.
    """

    def __init__(self, params, parameter_filename, skip_true_value_calc : bool = False):
        """
        Sets up the Chemical Equilibrium model class with model parameters.

        Parameters:  
            params : dict               Parameters for the model.
            parameter_filename : str    Filename and path of the file.   
            skip_true_value_calc : bool Skip evaulation of the true value (if not doing SPRE)   
        Returns:
            None         
        """

        # Call Parent’s constructor to set parameters
        super().__init__(params, parameter_filename)

        # Set initial SIR model
        self.model_name = "ChemEquil"
        self.xlabel = 'time'
        self.ylabel = 'concentration'
        self.title = 'Fast Chemical Equilibrium with Time-Varying Forcing'
        self.solution_labels = ["intermediate species", "product species"]

        # Set true value
        if not skip_true_value_calc:
            self.set_true_value()

    def diff_model(self, t : float, y : tuple) -> tuple:
        """
        Chemical equilibrium model. Returns current gradients of x and y

        Parameters:  
            t : float            Current time
            vals : tuple         Current values of x1 and x2 R stored as a tuple
        Returns:
            tuple                Current gradients
        """

        x1, x2 = y
        dx1dt = 0.5/x1
        dx2dt = 1.5*x1
    
        return (dx1dt, dx2dt)
    
    def get_initial_condition(self) -> tuple:
        """
        Returns initial condition of the Chem Equil model.

        Parameters:  
            None
        Returns:
            tuple                Initial values of model variables.
        """

        x1_0 = 1
        x2_0 = 1
        return (x1_0, x2_0)

    def set_true_value(self):
        """
        Sets the "true_value" of the Chem Equil model using analytic solution.
        
        Parameters:  
            None
        Returns:
            None                
        """
         
        self.true_value = np.pow(self.total_time + 1, 3/2)

    def get_final_quantity(self, discrete_paras : npt.NDArray) -> float:
        """
        Returns the final evaluation of the Chem Equil model given by last value of x2.

        Parameters:  
            discrete_paras : npt.NDArray        Discretisation parameters
        Returns:
            float                          
        """

        x1, x2 = self.diff_solution.y
        # Return final product species
        return x2[-1]

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
        filename += "_".join(str(i) for i in discrete_paras) + ".bin"

        return filename
    
    def set_true_value(self):
        """
        Sets the "true_value" of the Chem Equil model using analytic solution.
        
        Parameters:  
            None
        Returns:
            None                
        """

        # Get value at (0, 0) using analytic solution.
        self.true_value = self.diffusion_solution_2d(0, 0, self.total_time) 

    def get_final_quantity(self, discrete_paras : npt.NDArray) -> float:
        """
        Returns the final evaluation of the diffusion model given by value at (0, 0).

        Parameters:  
            discrete_paras : npt.NDArray        Discretisation parameters
        Returns:
            float                          
        """

        # Return value at (0,0) from simulated model.      
        return self.result.interpolate([0, 0])  
    
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


class PhysicsMugModel(Model):
    """
    Class for Physics model of a mug falling on a surface.
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
     
        self.total_time = 5.0

        # To decided when have objects stopped moving
        self.velocity_thresh = 1e-1
        self.steps_required_to_stop = 30

        # Use model f_z(x) = f(z+x)
        self.use_offset_model = False

        # Set default camera parameters
        self.camera_distance = 0.5                # closer to the object (default ~1.5)
        self.camera_yaw = 45                      # rotate horizontally
        self.camera_pitch = -50                   # angle downward
        self.camera_target_position = [0, 0, 0]
        # Scalar factor to delay mp4 at each step
        self.mp4_delay = 1

        # Mug settings
        self.object = "objects/mug.urdf"
        self.mug_linear_velocity = [0, 0, 0]
        self.mug_angular_velocity = [3.0, -1.5, 5.0] 
        self.base_position = [0, 0, 2.0]        # start above ground
        self.base_orientation = [0, 0, 0, 1]
        self.global_scaling = 1.0 
        
        # Call Parent’s constructor to set parameters
        super().__init__(params, parameter_filename)

        # Set initial model name
        self.model_name = "PhysicsMug"
        self.description = "Physics Mug Model"

        # Set default model parameters
        if self.final_mp4_filename is not None and self.final_mp4_filename != "":
            self.save_animation = True
            self.do_final_model_plot = True
        else:
            self.save_animation = False
       
        # Set true value
        if not self.save_animation and not skip_true_value_calc:
            self.set_true_value()
       
    def setup_model_world(self, dt : float, substeps : int, solver_iters : int, mp4_mode : bool = False) -> int:
        """
        Sets up world in pybullet to simulate model

        Parameters:  
            dt : float            Time step taken at each iteration in simulation
            substeps : int        Subdivide the physics simulation step further by 'numSubSteps'.
                                  This will trade performance over accuracy.
            solver_iters : int    The maximum number of constraint solver iterations. If the
                                  solverResidualThreshold (default, 1e-7) is reached, the solver may terminate before the numSolverIterations.  
            mp4_mode : bool       Use mode for creating mp4     
        Returns:
            int   
        """

        mode = pybullet.GUI if mp4_mode else pybullet.DIRECT

        # Set up physics simulator
        physicsClient = pybullet.connect(mode)

        # Add path for objects
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Simulation parameters
        pybullet.setPhysicsEngineParameter(
            fixedTimeStep = dt,
            numSubSteps = substeps,
            numSolverIterations = solver_iters
        )

        # Set the ground plane
        plane = pybullet.loadURDF("plane.urdf")

        # Create a simple mug object or some other object
        mug_id = pybullet.loadURDF( 
            self.object,     
            basePosition = self.base_position,        # start above ground
            baseOrientation = self.base_orientation,
            globalScaling = self.global_scaling
        )

        # Set gravity in world
        pybullet.setGravity(0, 0, -9.81)

        # Add initial spin to mug
        pybullet.resetBaseVelocity(
            mug_id,
            angularVelocity = self.mug_angular_velocity,  # spin around x, y, z
            linearVelocity = self.mug_linear_velocity
        )
       
        return mug_id

    def is_body_at_rest(self, body_id : object, do_it) -> bool:
        """
        Sets up world in pybullet to simulate model

        Parameters:  
            body_id : int              The object (mug) ID to check if stationary
            threshold : float          Threshold to check if it is stationary
        Returns:
            object   
        """

        # Get linear and angular velocities
        lin_vel, ang_vel = pybullet.getBaseVelocity(body_id)
        if do_it:
            print(lin_vel, ang_vel)
        return all(np.abs(lin_vel) < self.velocity_thresh) and all(np.abs(ang_vel) < self.velocity_thresh)

    def run_model_simulation(self, discrete_paras):
        """
        Uses pybullet package to simulate a falling spinning mug onto a surface.
        https://pybullet.org/wordpress/

        Parameters:  
            discrete_paras : npt.NDArray     Discretisation parameters used to simulate model.           
        Returns:
            float   
        """
   
        # Set discretisation parameters
        dt = discrete_paras[0]
        substeps = int(np.round(1.0/discrete_paras[1]))
        solver_iters = int(np.round(1.0/discrete_paras[2]))

        mug = self.setup_model_world(dt, substeps, solver_iters)
      
        # Output info on what is being simulated
        print(f"\tSimulating {self.description} with dt = {dt}, {substeps} substeps and {solver_iters} solver iterations")
 
        # Initial time counter and stationary counter
        sim_time = 0.0
        stationary_count = 0

        # Run the simulation
        while sim_time < self.total_time:
            # One step of simulation
            pybullet.stepSimulation()
            sim_time += dt   
            #if sim_time > 550:         
            #    print(sim_time) 
            # Now stop if the mug (or object) is stationary
            if self.is_body_at_rest(mug, False):
                stationary_count += 1
                if stationary_count >= self.steps_required_to_stop:                    
                    break
            else:
                stationary_count = 0

        print("\tEnd time:", sim_time)

        # Get final position and orientation of mug
        pos, orn = pybullet.getBasePositionAndOrientation(mug)

        # Get distance of mug from origin
        dist = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
       
        # End simulation
        pybullet.disconnect()

        # Do video if req'd
        if self.save_animation:
            self.record_mp4()

        print(f"\tCalculated final value: {dist}")
        return dist

    def plot_final_model(self):
        """
        Plots the final simulated model which is in this case is a mp4 video if req'd.

        Parameters:  
            None
        Returns:
            None                
        """
   
        if self.save_animation:
            self.record_mp4()

    def record_mp4(self):
        """
        Create mp4 video of the model of a falling spinning mug onto a surface.
       
        Parameters:  
            None         
        Returns:
            None
        """

        # Set discretisation parameters
        dt = self.final_tols[0]
        substeps = int(np.round(1.0/self.final_tols[1]))
        solver_iters = int(np.round(1.0/self.final_tols[2]))

        # Output info on what is being simulated
        print(f"\tSimulating {self.description} for mp4 with dt = {dt}, {substeps} substeps and {solver_iters} solver iterations")

        self.setup_model_world(dt, substeps, solver_iters, True)

        # Zoomed-in camera settings
        pybullet.resetDebugVisualizerCamera(
            cameraDistance = self.camera_distance,                # closer to the cube (default ~1.5)
            cameraYaw = self.camera_yaw,                      # rotate horizontally
            cameraPitch = self.camera_pitch,                   # angle downward
            cameraTargetPosition = self.camera_target_position   # look at where the cube will fall
        )
 
        # Hide GUI
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        # Start video recording
        log_id = pybullet.startStateLogging(
            pybullet.STATE_LOGGING_VIDEO_MP4,
            self.final_mp4_filename
        )
            
        # Initial time counter
        sim_time = 0.0

        while sim_time < self.total_time:
            # One step of simulation
            pybullet.stepSimulation()

            # Make real-time video look normal - but only if dt ~= 1/240 - needs updating otherwise  
            time.sleep(dt * self.mp4_delay) # fudge factor

            sim_time += dt
  
        # Stop filming mug
        pybullet.stopStateLogging(log_id)
        print(f"Saved video to {self.final_mp4_filename}")

        # End simulation
        pybullet.disconnect()

    
    def get_cache_filename(self, discrete_paras : npt.NDArray):
        """
        Returns the model cache filename for the Physics Mug model based on the model parameters.

        Parameters:  
            discrete_paras : npt.NDArray   Discretisation parameters
        Returns:
            str    
        """
      
        # Create filename with all settings and parameters used
        filename = f"pm_{self.total_time}_"
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

        # Use regular model if evaluation the offset model, f_z(x) = f(z+x). So evaluate f_z(0) = f(z)
        use_offset_model = self.use_offset_model
        self.use_offset_model = False

        self.true_value = self.run_model(self.final_tols)

        # Set back as before
        self.use_offset_model = use_offset_model

class PhysicsDuckModel(PhysicsMugModel):
    """
    Class for Physics model of a rubber duck falling a tiny bit and coming to rest on a surface.
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
     
        # Model parameters that can be updated in scenario file
        self.total_time = 5.0
      
        # Call Parent’s constructor to set parameters
        super().__init__(params, parameter_filename, skip_true_value_calc = True)

        # Set default camera parameters
        self.camera_distance = 0.5                # closer to the object (default ~1.5)
        self.camera_yaw = 45                      # rotate horizontally
        self.camera_pitch = -50                   # angle downward
        self.camera_target_position = [0, 0, 0]

        # Mug settings
        #self.object = "duck_vhacd.urdf"
        self.mug_angular_velocity = [3.0, -1.5, 5.0] 
        self.base_position = [0, 0, 1.0]        # start above ground
        self.base_orientation = [0, 0, 0, 1]
        self.global_scaling = 1.0 
       
        # Set initial model name
        self.model_name = "PhysicsDuck"
        self.description = "Physics Rubber Duck Model"

         # Set default model parameters
        if self.final_mp4_filename is not None and self.final_mp4_filename != "":
            self.save_animation = True
            self.do_final_model_plot = True
        else:
            self.save_animation = False
       
        # Set true value
        if not self.save_animation and not skip_true_value_calc:
            self.set_true_value()
    
class PhysicsQuickModel(PhysicsMugModel):
    """
    Class for Physics model of some object in a quick scenerio.
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
     
       
        self.total_time = 5.0
      
        # Call Parent’s constructor to set parameters
        super().__init__(params, parameter_filename, skip_true_value_calc = True)

        # Parameters set unchangable for this model below
        # Set default camera parameters
        self.camera_distance = 0.2                # closer to the object (default ~1.5)
        self.camera_yaw = 45                      # rotate horizontally
        self.camera_pitch = -50                   # angle downward
        self.camera_target_position = [0, 0, 0]

        # Object settings
        #self.object = "domino/domino.urdf"
        self.object = "lego/lego.urdf"
        self.mug_angular_velocity = [0.0, 0.0, 0.0] 
        self.base_position = [0, 0, 1.0]        # start above ground
        self.base_orientation = [0.2, -0.1, 0.05, 1]
        self.global_scaling = 1.0 
        
        # Set initial model name
        self.model_name = "PhysicsQuick"
        self.description = "Physics Quick Model"

         # Set default model parameters
        if self.final_mp4_filename is not None and self.final_mp4_filename != "":
            self.save_animation = True
            self.do_final_model_plot = True
        else:
            self.save_animation = False

        # Set true value
        if not self.save_animation and not skip_true_value_calc:
            self.set_true_value()

class PhysicsSlickModel(PhysicsMugModel):
    """
    Class for Physics model of some object in a slick scenerio.
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
     
        self.total_time = 2.0
      
        # Call Parent’s constructor to set parameters
        super().__init__(params, parameter_filename, skip_true_value_calc = True)

        # Parameters set unchangable for this model below
        # Set default camera parameters
        self.camera_distance = 2.0                # closer to the object (default ~1.5)
        self.camera_yaw = 45                      # rotate horizontally
        self.camera_pitch = -50                   # angle downward
        self.camera_target_position = [0, 0, 0]

        # Object settings
        #self.object = "domino/domino.urdf"
        self.object = "soccerball.urdf"
        self.mug_linear_velocity = [-1.0, 0.0, 0]
        self.mug_angular_velocity = [-1.0, 0.0, 0.0] 
        self.base_position = [0, 0, 0.55]        # start above ground
        self.base_orientation = [0.2, -0.1, 0.05, 1]
        self.global_scaling = 1.0 
        
        # Set initial model name
        self.model_name = "PhysicsSlick"
        self.description = "Physics Slick Model"

         # Set default model parameters
        if self.final_mp4_filename is not None and self.final_mp4_filename != "":
            self.save_animation = True
            self.do_final_model_plot = True
        else:
            self.save_animation = False

        # Set true value
        if not self.save_animation and not skip_true_value_calc:
            self.set_true_value()   

class MujocoPhysicsModel(Model):
    """
    Class for Physics model using the MuJoCo (Multi-Joint dynamics with Contact) python library.
    https://mujoco.readthedocs.io/
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
     
        self.total_time = 5.0

        # Use model f_z(x) = f(z+x)
        self.use_offset_model = False
        self.substeps_max = None

        # Set default camera parameters
        self.camera_position = np.array([2, -2, 1.5])
        # Smaller is closer to the object
        self.camera_distance_scale = 0.5                
        self.fps = 60

        # Set initial model description       
        self.description = "MuJoCo Physics Model"

        # Call Parent’s constructor to set parameters - and overwrite any above here but not below
        super().__init__(params, parameter_filename)

        # Set initial model name    
        self.model_name = "MujocoPhysics"

        # Set default model parameters
        if self.final_mp4_filename is not None and self.final_mp4_filename != "":
            self.save_animation = True
            self.do_final_model_plot = True
        else:
            self.save_animation = False
       
        # Set true value
        if not self.save_animation and not skip_true_value_calc:
            self.set_true_value()
     
    def setup_model_world(self, dt : float, substeps : int):
        """
        Sets up world in MuJoCo to simulate model.

        Parameters:  
            dt : float          Timestap
            substeps : int      Number of substeps      
        Returns:
            None  
        """

        # Set camera zoom
        camera_pos = self.camera_position * self.camera_distance_scale

        # Read MJCF from file, XML file with the model setup
        # Set the directory of the file
        model_file = str(Path(self.cache_dir).parent.parent / self.model_file)
        with open(model_file, "r") as f:            
            mjcf_str = f.read()

        # Insert timestep, impratio and camera position into MJCF
        mjcf = mjcf_str.format(timestep=dt, impratio=substeps, camera_pos_x=camera_pos[0], camera_pos_y=camera_pos[1], camera_pos_z=camera_pos[2])

        # Load model and data
        self.model = mujoco.MjModel.from_xml_string(mjcf)
        self.data = mujoco.MjData(self.model)

        # Give the sphere an initial velocity for angled impact
        # qvel layout for a free joint: [vx, vy, vz, wx, wy, wz]
        self.data.qvel[:3] = np.array([0.1, 0.1, 0.0]) 

    def run_model_simulation(self, discrete_paras):
        """
        Uses MuJoCo (Multi-Joint dynamics with Contact) Python library to simulate scenario as given in XML setup file.
        https://mujoco.readthedocs.io/

        Parameters:  
            discrete_paras : npt.NDArray     Discretisation parameters used to simulate model.           
        Returns:
            float   
        """
   
        # Set discretisation parameters
        dt = discrete_paras[0]
        # Set substeps
        if self.substeps_max is not None:
            substeps = int(self.substeps_max * (1.0 - discrete_paras[1]))
        else:
            substeps = int(np.round(1.0/discrete_paras[1]))
       
        # Output info on what is being simulated
        print(f"\tSimulating {self.description} with dt = {dt} and {substeps} substeps iterations")
 
        # Setup model world
        self.setup_model_world(dt, substeps)

        # Renderer
        renderer = mujoco.Renderer(self.model, width=640, height=480)
        self.frames = []

        # Truncates the number of steps, so try and make scenario come to rest to avoid edge effects.
        steps = int(self.total_time / self.model.opt.timestep)
        frame_interval = int(1.0 / (self.fps * self.model.opt.timestep))

        for step in range(steps):
            mujoco.mj_step(self.model, self.data)

            if self.save_animation and self.step % frame_interval == 0:
                renderer.update_scene(self.data, camera="angled_view")
                frame = renderer.render()
                self.frames.append(frame)

        # Final position & distance
        body_id = self.model.body("sphere").id
        final_position = self.data.xpos[body_id].copy()
        distance = np.linalg.norm(final_position)

        print(f"\tCalculated final value: {distance}")
        return distance

    def plot_final_model(self):
        """
        Plots the final simulated model which is in this case is a mp4 video if req'd.

        Parameters:  
            None
        Returns:
            None                
        """
   
        if self.save_animation:
            imageio.mimsave(self.final_mp4_filename, self.frames, fps=self.fps)

    def get_cache_filename(self, discrete_paras : npt.NDArray):
        """
        Returns the model cache filename for the Physics Mug model based on the model parameters.

        Parameters:  
            discrete_paras : npt.NDArray   Discretisation parameters
        Returns:
            str    
        """
      
        if self.substeps_max is not None:
            substeps_max_str = f"{self.substeps_max}_"
        else:
            substeps_max_str = ""

        # Create filename with all settings and parameters used
        filename = f"mp_{self.total_time}_{self.model_file[:-4]}_" + substeps_max_str
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

        # Use regular model if evaluation the offset model, f_z(x) = f(z+x). So evaluate f_z(0) = f(z)
        use_offset_model = self.use_offset_model
        self.use_offset_model = False

        self.true_value = self.run_model(self.final_tols)

        # Set back as before
        self.use_offset_model = use_offset_model

class MujocoModel(Model):
    """
    Class for Physics model using the MuJoCo (Multi-Joint dynamics with Contact) python library.
    https://mujoco.readthedocs.io/
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
     
        self.total_time = 600.0
        # To decided when have objects stopped moving
        self.velocity_thresh = 1e-15
        self.steps_required_to_stop = 30

        # Use model f_z(x) = f(z+x)
        self.use_offset_model = False
      
        # Set default camera parameters
        self.camera_position = np.array([2, -2, 1.5])
        # Smaller is closer to the object
        self.camera_distance_scale = 0.5                
        self.fps = 60

        # Set initial model description       
        self.description = "MuJoCo Physics Model"

        # Call Parent’s constructor to set parameters - and overwrite any above here but not below
        super().__init__(params, parameter_filename)

        # Set initial model name    
        self.model_name = "Mujoco"

        # Set default model parameters
        if self.final_mp4_filename is not None and self.final_mp4_filename != "":
            self.save_animation = True
            self.do_final_model_plot = True
        else:
            self.save_animation = False
       
        # Set true value
        if not self.save_animation and not skip_true_value_calc:
            self.set_true_value()
     
    def setup_model_world(self, dt : float, tolerance : float):
        """
        Sets up world in MuJoCo to simulate model.

        Parameters:  
            dt : float          Timestap
            tolerance : float   Tolerance of solver
        Returns:
            None  
        """

        # Set camera zoom
        camera_pos = self.camera_position * self.camera_distance_scale

        # Read MJCF from file, XML file with the model setup
        # Set the directory of the file
        model_file = str(Path(self.cache_dir).parent / self.model_file)
      
        with open(model_file, "r") as f:            
            mjcf_str = f.read()

        # Insert timestep, impratio and camera position into MJCF
        mjcf = mjcf_str.format(timestep=dt, tolerance=tolerance, camera_pos_x=camera_pos[0], camera_pos_y=camera_pos[1], camera_pos_z=camera_pos[2])

        # Load model and data
        self.model = mujoco.MjModel.from_xml_string(mjcf)
        self.data = mujoco.MjData(self.model)

        # Set initial velocities for objects
        for body_id in range(self.model.nbody):

            # How many joints does this body have?
            njnt = self.model.body_jntnum[body_id]
            if njnt == 0:
                continue  # static body (e.g. floor)

            # First joint index for this body
            jntid = self.model.body_jntadr[body_id]

            # DOF start index in qvel
            dofadr = self.model.jnt_dofadr[jntid]

            # Read velocities from XML user field
            user_vals = self.model.body_user[body_id]

            if user_vals is not None and len(user_vals) >= 6:
                self.data.qvel[dofadr:dofadr+6] = np.array(user_vals[:6])

    def run_model_simulation(self, discrete_paras):
        """
        Uses MuJoCo (Multi-Joint dynamics with Contact) Python library to simulate scenario as given in XML setup file.
        https://mujoco.readthedocs.io/

        Parameters:  
            discrete_paras : npt.NDArray     Discretisation parameters used to simulate model.           
        Returns:
            float   
        """
   
        # Set discretisation parameters
        dt = discrete_paras[0]
        tolerance = discrete_paras[1]

        # Output info on what is being simulated
        print(f"\tSimulating {self.description} with dt = {dt} and tolerance = {tolerance}")
 
        # Setup model world
        self.setup_model_world(dt, tolerance)

        # Total time is used as an upper limit all objects should come to rest well before this
        steps = int(self.total_time / self.model.opt.timestep)

        # Set up if creating a video
        if self.save_animation:
            # Renderer
            renderer = mujoco.Renderer(self.model, width=640, height=480)
            self.frames = []
            frame_interval = int(1.0 / (self.fps * self.model.opt.timestep))
            if frame_interval == 0:
                frame_interval = 1
                print("Warning: frame interval too small, set a smaller time step!")

        # Get all body IDs, only include bodies with joints (movable bodies)
        body_ids = [i for i in range(self.model.nbody) if self.model.body_jntadr[i] != -1]
        stop_steps = 1

        for step in range(steps):
            mujoco.mj_step(self.model, self.data)

            # Save frames for video if creating one
            if self.save_animation and step % frame_interval == 0:
                renderer.update_scene(self.data, camera="angled_view")
                frame = renderer.render()
                self.frames.append(frame)

            # Check if everything has stopped
            stop_sim = True
            for body_id in body_ids:
                stop_sim = stop_sim and all(np.abs(self.data.cvel[body_id]) < self.velocity_thresh)

            # Stop if stationary for a number of steps
            if stop_sim:
                stop_steps += 1
                if stop_steps >= self.steps_required_to_stop:
                    break

        print("\tStop time: ", step*self.model.opt.timestep)
        for body_id in body_ids:
                print(np.abs(self.data.cvel[body_id]))

        # Final position & distance
        total_distance = 0.0
        for body_id in body_ids:          
            pos = self.data.xpos[body_id]  # world position of body
            distance = np.linalg.norm(pos)
            total_distance += distance

        print(f"\tCalculated final value: {total_distance}")
        return total_distance

    def plot_final_model(self):
        """
        Plots the final simulated model which is in this case is a mp4 video if req'd.

        Parameters:  
            None
        Returns:
            None                
        """
        
        if self.save_animation:            
            imageio.mimsave(self.final_mp4_filename, self.frames, fps=self.fps)

    def get_cache_filename(self, discrete_paras : npt.NDArray):
        """
        Returns the model cache filename for the Physics Mug model based on the model parameters.

        Parameters:  
            discrete_paras : npt.NDArray   Discretisation parameters
        Returns:
            str    
        """
      
        # Create filename with all settings and parameters used
        filename = f"mp_{self.total_time}_{self.model_file[:-4]}_"

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

        # Use regular model if evaluation the offset model, f_z(x) = f(z+x). So evaluate f_z(0) = f(z)
        use_offset_model = self.use_offset_model
        self.use_offset_model = False

        self.true_value = self.run_model(self.final_tols)

        # Set back as before
        self.use_offset_model = use_offset_model
