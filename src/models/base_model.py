##############################################################################
# Simulation Models parent class
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import numpy as np
import numpy.typing as npt
import os
import struct
from pathlib import Path
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Application modules
from sparse_pre.extrapolation import extrapolation

# For default fonts
import matplotlib as mpl
mpl.rcdefaults()

class Model:
    """
    Base model class with common methods used for all model classes.
    """

    def __init__(self, params : dict, parameter_filename : str, skip_true_value_calc : bool = False) -> None:
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
        self.use_model_cache = False
        self.use_fixed_basis = False
        self.max_order = 20

        # Set filenames to blank by default
        self.results_plot_filename = ""
        self.results_filename = ""
        self.results_plot_filename = ""
        self.initial_model_plot_filename = ""
        self.final_model_plot_filename = ""
        self.final_mp4_filename = ""
        self.screenshot_filename = ""
        self.results_fx_filename = ""
        self.results_bases_filename = ""

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
    
    def set_parameters(self, parameters : dict) -> None:
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

        if "initial_model_plot_filename" not in parameters.keys():
            self.initial_model_plot_filename = None

        if "final_model_plot_filename" not in parameters.keys():
            self.final_model_plot_filename = None
        
        if "final_mp4_filename" not in parameters.keys():
            self.final_mp4_filename = None

        if "screenshot_filename" not in parameters.keys():
            self.screenshot_filename = None

        if "results_fx_filename" not in parameters.keys():
            self.results_fx_filename = None
        
        if "results_bases_filename" not in parameters.keys():
            self.results_bases_filename = None

    # Files to save results
    def add_path(self, path : str, filename : str) -> None:
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
        
    def update_paths(self, parameter_filename : str) -> None:
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
        self.initial_model_plot_filename = self.add_path(results_dir, self.initial_model_plot_filename)
        self.final_model_plot_filename = self.add_path(results_dir, self.final_model_plot_filename)
        self.final_mp4_filename = self.add_path(results_dir, self.final_mp4_filename)
        self.screenshot_filename = self.add_path(results_dir, self.screenshot_filename)
        self.results_fx_filename = self.add_path(results_dir, self.results_fx_filename)
        self.results_bases_filename = self.add_path(results_dir, self.results_bases_filename)

        self.do_results_plot = self.results_plot_filename != ""
        self.do_final_model_plot = self.final_model_plot_filename != ""

        if self.evaluation:
            self.results_eval_filename = self.add_path(results_dir, self.results_eval_filename)
            self.results_eval_plot_filename = self.add_path(results_dir, self.results_eval_plot_filename)
            self.do_results_eval_plot = self.results_eval_plot_filename != ""

    def set_true_value(self) -> None:
        """
        Sets the object variable "true_value" to the actual model outcome.
        For example, from an analytic solution if known. This can be used to evaluate model accuracy.

        By default sets the "true_value" by running the model with small discretisation parameters
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

    def get_cache_filename(self, discrete_paras : npt.NDArray) -> str:
        """
        Returns the model cache filename based on the model parameters.

        Parameters:  
            discrete_paras : npt.NDArray   Discretisation parameters
        Returns:
            str    
        """

        return self.model_name + "_".join(str(i) for i in discrete_paras) + ".bin"

    def update_model_cache(self, discrete_paras : npt.NDArray, y : float) -> None:
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
            if self.use_model_cache:
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
     
    def run_analysis(self) -> None:
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
            "use_fixed_basis": self.use_fixed_basis,
            "bases_filename": self.results_bases_filename,
            "max_order": self.max_order
        }

        # Remove bases file if it exists so that it can be appended to later
        if self.results_bases_filename != "":
            if os.path.exists(self.results_bases_filename):
                os.remove(self.results_bases_filename)

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
            
            out = extrapolation(X*h, Y, options, h)

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

            # Create table of absolute errors
            if self.evaluation:
                # h may itself be a vector (e.g. dt, delta, ...)
                if isinstance(h, (list, tuple, np.ndarray)):
                    table_row = list(h)
                else:
                    table_row = [h]

                # Append:                     
                values = [np.asarray(self.true_value).reshape(1), Y, np.asarray(out['mu'][0]).reshape(1), np.abs(self.true_value - Y), np.asarray(np.abs(self.true_value - out['mu'][0])).reshape(1)]
                # Flatten list of lists
                values_to_add = np.array(np.concatenate(values).tolist())
                table_row.extend(values_to_add)
                
                table_row = np.asarray(table_row, dtype=float)

                if i == 0:
                    self.abs_error_table = table_row.reshape(1, -1)
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

    def simulate_ith_analysis_setting(self, sim_number : int) -> None:
        """
        Simulates model for the set up parameters for the ith 
        discretisation parameters given by X and h arrays. The simulation outcome will then
        be added to the cache which can be used in subsequent analyses - if the cache is set for use.
   
        Parameters:  
            sim_number : int
        Returns:
            None                
        """

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

    def plot_SPRE_results(self) -> None:
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
    
        plt.tight_layout()
        plt.savefig(self.results_plot_filename) 
        plt.show()    

    def plot_final_model(self) -> None:
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

    def plot_evaluation_results(self) -> None:
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

        # -------------------------------
        # Build DataFrame of absolute errors
        # -------------------------------

        # Header for h
        if not isinstance(self.h_values[0], (list, tuple)):
            abs_header = ["h"]
        else:
            abs_header = [f"h{i+1}" for i in range(len(self.h_values[0]))]

        n_est = np.array(self.X).shape[0]  # number of estimates per run

        # Full header
        abs_header += (
            ["true_value"]
            + [f"estimate_{i+1}" for i in range(n_est)]
            + ["spre_estimate"]
            + [f"abs_err_estimate_{i+1}" for i in range(n_est)]
            + ["abs_err_spre_estimate"]
        )

        # Create DataFrame
        df_abs = pd.DataFrame(self.abs_error_table, columns=abs_header)

        # -------------------------------
        # Choose x-axis values
        # -------------------------------

        h_col = self.choose_h_column(df_abs)
        x_vals = df_abs[h_col]
        x_lab = "h"

        if hasattr(self, "eval_plot_type") and self.eval_plot_type == 2:
            x_vals = 2.0 / df_abs[abs_header[2]]
            x_lab = "grid spacing"

        # -------------------------------
        # Save results to file
        # -------------------------------

        if self.results_eval_filename:
            df_abs.to_csv(self.results_eval_filename, sep="\t", index=False)

        # -------------------------------
        # Plot absolute errors
        # -------------------------------

        if self.do_results_eval_plot:

            plt.close("all")
            plt.figure(figsize=(7, 6))

            # Raw estimates (faded)
            for i in range(n_est):
                plt.plot(
                    x_vals,
                    df_abs[f"abs_err_estimate_{i+1}"],
                    marker="o",
                    linewidth=2,
                    markersize=8,
                    alpha=0.4,
                    label="raw estimates" if i == 0 else None
                )

            # SPRE estimates (highlighted)
            plt.plot(x_vals, df_abs["abs_err_spre_estimate"], marker='o', linestyle='solid', linewidth=2, markersize=12, label=f"{self.extrapolation_name} estimate", color ="black")

          
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel(x_lab)
            plt.ylabel("absolute error")
            plt.title("Absolute Errors of f(0) Estimates")
            plt.grid(True, which="both", linestyle="--", alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.results_eval_plot_filename)
            plt.show()

