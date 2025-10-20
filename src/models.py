##############################################################################
# Differential equation models
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import numpy as np
import numpy.typing as npt
import json
import os
from pathlib import Path
import pandas as pd
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Application modules
from sparse_pre.extrapolation import extrapolation

class Model:
    """
    Base model class
    """

    def __init__(self, params : dict, parameter_filename : str):
        # Set default parameters values  
        self.total_time = 120
        self.evaluation = False
        self.model_name = "Model not set"

        # Set parmaters
        self.set_parameters(params)

        # Update paths for result files and plots
        self.update_paths(parameter_filename)

        # Set true value
        self.set_true_value()

        # Model labels
        self.xlabel = 'time'
        self.ylabel = 'y'
        self.title = 'Model'
        
    def __str__(self):
        return self.model_name
    
    def diff_model(self, t : float, y : tuple) -> tuple:
        return (0, 0)
       
    def get_initial_condition(self) -> tuple:
        return (0, 0)
    
    def get_final_quantity(self, discrete_paras : npt.NDArray) -> float:
        return 0
    
    def set_parameters(self, parameters : dict):
        
        """Set variables from a dictionary."""
        for key, value in parameters.items():
            setattr(self, key, value)

        # Check if running evalution of SPRE method
        if "evaluation" not in parameters.keys():
            self.evaluation = False

        if "final_model_plot_filename" not in parameters.keys():
            self.final_model_plot_filename = None

    # Files to save results
    def add_path(self, path : str, filename : str):
        new_filename = ""
        if filename is not None and filename:
            new_filename = os.path.join(path, filename)
        return new_filename
        
    def update_paths(self, parameter_filename : str):
        
        # Get the directory of the input file
        write_dir = os.path.dirname(parameter_filename)
  
        self.results_filename = self.add_path(write_dir, self.results_filename)
        self.results_plot_filename = self.add_path(write_dir, self.results_plot_filename)
        self.final_model_plot_filename = self.add_path(write_dir, self.final_model_plot_filename)

        self.do_results_plot = self.results_plot_filename != ""
        self.do_final_model_plot = self.final_model_plot_filename != ""

        if self.evaluation:
            self.results_eval_filename = self.add_path(write_dir, self.results_eval_filename)
            self.results_eval_plot_filename = self.add_path(write_dir, self.results_eval_plot_filename)
            self.do_results_eval_plot = self.results_eval_plot_filename != ""

    def set_true_value(self):
        self.true_value = 0

    def run_model(self, discrete_paras : npt.NDArray) -> float:
        """
        Runs model by solving diff equations   
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
             
        # Values to try
        X = np.array(self.X)

        # Apply SPRE
        # Define options
        # "name"   : str, one of {"MRE", "GRE", "SPRE"} (default: "SPRE")
        # "k_name" : str, one of {"Gaussian", "GaussianARD", "Matern1/2", "Matern3/2", "white"} (default: "white")
        options = {
            "name": self.extrapolation_name,
            "k_name":  self.extrapolation_kernel, 
            "plot" : False
        }

        # Loop thro' different scalar values for multiplying set of tolerences
        for i, h in enumerate(self.h_values):
            # Results, Y is model output
            Y = np.array([])
            extrapolation_results = []

            # Get results
            for x in X:                                            
                y = self.run_model(h * x)
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

            out = extrapolation(X, Y, options)
            print(f"Predict f(0) = {out['mu'][0]} +/- {np.sqrt(out['var'][0][0])}\n")
            
            extrapolation_results.extend([h, out['mu'][0], out['var'][0][0]])

            # Append results for each point
            extrapolation_results.extend(out['mu_cv'])
            extrapolation_results.extend(out['var_cv'])
        
            # Create table of SPRE results
            if i == 0:
                all_extrapolation_results = np.matrix(extrapolation_results)
            else:
                all_extrapolation_results = np.vstack((all_extrapolation_results, extrapolation_results))

            # Create table of absoluate errors
            if self.evaluation:
                # Create row of results for absolute error table
                # h, true_value, best_estimate, spre_estimate, abs_err_best_estimate, abs_err_spre_estimate
                table_row = np.array([h, self.true_value, Y[0], out['mu'][0], np.abs(self.true_value - Y[0]), np.abs(self.true_value - out['mu'][0])])

                if i == 0:
                    self.abs_error_table = np.matrix(table_row)
                else:
                    self.abs_error_table = np.vstack((self.abs_error_table, table_row))

            # Create dataframe of results
            number_of_x = X.shape[0]
            header = ["h", "mu", "var"] + [f"mu_cv{n}" for n in range(1, number_of_x + 1)] + [f"var_cv{n}" for n in range(1, number_of_x + 1)]
  
            # Create DataFrame
            self.df_all_extrapolation_results = pd.DataFrame(all_extrapolation_results, columns=header)

            # Write results to file
            if self.results_filename:
                # Write to file with tab separation
                self.df_all_extrapolation_results.to_csv(self.results_filename, sep="\t", index=False)

        
        # Do plots for the analysis if requested
        if self.evaluation:
           self.plot_evaulation_results() 

        else:
            if self.do_results_plot:
                self.plot_SPRE_results()
                
        if self.do_final_model_plot:
            _ = self.plot_diff_solution()


    def plot_SPRE_results(self):
        # Calculate error bars (standard deviation = sqrt(var))
        errors = np.sqrt(self.df_all_extrapolation_results["var"])

        # Plot with error bars
        plt.close('all') 
        plt.figure()
        plt.errorbar(self.df_all_extrapolation_results["h"], self.df_all_extrapolation_results["mu"], yerr=errors, fmt='o-', capsize=5, ecolor='black', markersize=6)
        plt.xlabel("h")
        plt.ylabel("mu")
        plt.title("Extrapolation Results")
        plt.grid(True)

        # Add horizontal dashed line with computed accurate answer
        if self.final_tols is not None: 
            
            print(f"\nTrue value calculated as f(0) = {self.true_value}\n")

            plt.axhline(y = self.true_value, color='red', linestyle='--', linewidth=1)
    
        plt.savefig(self.results_plot_filename) 
        plt.show()    

    def plot_diff_solution(self):

        # -----------------------------
        # Plot Results
        # -----------------------------
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

    def plot_evaulation_results(self):
        # Create table of absolute errors (wrt to "true" value) with
        # 1) Smallest time step in set of time steps
        # 2) SPRE estimate using all time steps in set

        #abs_error_table = np.vstack((all_extrapolation_results, extrapolation_results))

        # Create DataFrame
        abs_header = ["h", "true_value", "best_estimate", "spre_estimate", "abs_err_best_estimate", "abs_err_spre_estimate"]
        df_abs = pd.DataFrame(self.abs_error_table, columns=abs_header)

        # Write results to file
        if self.results_eval_filename:
            # Write to file with tab separation
            df_abs.to_csv(self.results_eval_filename, sep="\t", index=False)

        if self.do_results_eval_plot:
            # Plot absolute errors of two estimates

            plt.close('all') 
            plt.figure()
            plt.plot(df_abs["h"], df_abs["abs_err_best_estimate"], marker='o', linestyle='solid', linewidth=2, markersize=12, label="best estimate")
            plt.plot(df_abs["h"], df_abs["abs_err_spre_estimate"], marker='o', linestyle='solid', linewidth=2, markersize=12, label="SPRE estimate")
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("discretization parameter")
            plt.ylabel("absolute error")
            plt.title("Absolute Errors of f(0) Estimates")
            plt.grid(True)
            plt.legend()
            plt.savefig(self.results_eval_plot_filename) 
            plt.show()

def get_model(parameter_filename : str) -> Model:
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
    #model = getattr(diff_models, model_class_name)(params, parameter_filename)
    model = globals()[model_class_name](params, parameter_filename)

    return model

class SirModel(Model):
    def __init__(self, params, parameter_filename):
        # Call Parent’s constructor to set parameters
        super().__init__(params, parameter_filename)

        # Set initial SIR model
        self.model_name = "SIR"
        # Model labels
        self.xlabel = 'time (days)'
        self.ylabel = 'population'
        self.title = 'SIR Model'
        self.solution_labels = ["Susceptible", "Infected", "Recovered"]
        
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

        S0 = self.N - self.initial_infected
        I0 = self.initial_infected
        R0 = 0
        return (S0, I0, R0)

    def set_true_value(self):
        self.true_value = self.run_model(self.final_tols)

    def get_final_quantity(self, discrete_paras : npt.NDArray) -> float:

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
    def __init__(self, params, parameter_filename):
        # Call Parent’s constructor to set parameters
        super().__init__(params, parameter_filename)

        # Set initial SIR model
        self.model_name = "ChemEquil"
        self.xlabel = 'time'
        self.ylabel = 'concentration'
        self.title = 'Fast Chemical Equilibrium with Time-Varying Forcing'
        self.solution_labels = ["intermediate species", "product species"]

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
        x1_0 = 1
        x2_0 = 1
        return (x1_0, x2_0)

    def set_true_value(self):
        self.true_value = np.pow(self.total_time + 1, 3/2)

    def get_final_quantity(self, discrete_paras : npt.NDArray) -> float:
        x1, x2 = self.diff_solution.y
        # Return final product species
        return x2[-1]