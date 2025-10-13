##############################################################################
# This model describes a fast chemical or biochemical equilibrium in which one substance (x(t))
#  instantaneously adjusts to a slowly changing external condition (represented by time (t)),
#  while another quantity (y(t)) is produced from (x) through a simple stoichiometric relationship. 
#
# From root directory, for example run
# python .\src\chem_equil.py .\data\chem_equil\input_CHEQ_Eval_1.json
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d
import json
import sys
import os
import pandas as pd

# Application modules
from sparse_pre.extrapolation import extrapolation

def chem_equil_model(t : float, vals : tuple) -> tuple:
    """
    Chemical equilibrium model. Returns current gradients of x and y

    Parameters:  
        t : float            Current time
        vals : tuple         Current values of x1 and x2 R stored as a tuple
    Returns:
        tuple                Current gradients of S, I and R
    """

    x1, x2 = vals
    dx1dt = 0.5/x1
    dx2dt = 1.5*x1
   
    return (dx1dt, dx2dt)

def run_chem_equil_model(diff_tol : float = 1e-8, 
                  total_time : float = 120,
                  plot_filename : str = "", evaluation : bool = False):
    """
    Runs model for fast chemical equilibrium with time-varying forcing    
    """
   
    
    x1_0 = 1
    x2_0 = 1
    y0 = [x1_0, x2_0]

    # Time span to evalute the SIR model
    t_span = (0, total_time)

    # Timepoints at which to store values
    number_of_points = 100000 #max(2, int(1/diff_tol)) + 1
    t_eval = np.linspace(*t_span, number_of_points)

    # -----------------------------
    # Solve system together
    # -----------------------------
    sol = solve_ivp(
        fun=chem_equil_model,
        t_span=t_span,
        y0=y0,
        args=None,
        t_eval=t_eval,
        method='RK45',
        rtol=diff_tol  
    )

    x1, x2 = sol.y
    t = sol.t

    if plot_filename:
        # -----------------------------
        # Plot Results
        # -----------------------------
        plt.close('all') 
        plt.figure(figsize=(10, 6))
        plt.plot(t, x1, label='intermediate species')
        plt.plot(t, x2, label='product species')
      
        plt.xlabel('time')
        plt.ylabel('concentration')
        plt.title('Fast Chemical Equilibrium with Time-Varying Forcing')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(plot_filename)  
        plt.show()

    
    # Return final product species
    return x2[-1]
   

# ----------------------------------------------------------
# Read in parameters and options for running SPRE with various tolerences
# ----------------------------------------------------------
# Get new filename
parameter_filename = sys.argv[1]

# Get parameters
with open(parameter_filename) as f:
    params = json.load(f)

# Get the directory of the input file
write_dir = os.path.dirname(parameter_filename)

total_time = params["total_time"]
X = params["X"]
h_values = params["h_values"]
final_tols = params["final_tols"]

# Check if running evalution of SPRE method
if "evaluation" in params.keys():
    evaluation = params["evaluation"]
else:
    evaluation = False

# Files to save results
def add_path(path, filename):
    new_filename = ""
    if filename is not None and filename:
        new_filename = os.path.join(path, filename)
    return new_filename

results_filename = add_path(write_dir, params["results_filename"])
results_plot_filename = add_path(write_dir, params["results_plot_filename"])
final_model_plot_filename = add_path(write_dir, params["final_model_plot_filename"])
results_plot_filename = add_path(write_dir, params["results_plot_filename"])

do_results_plot = params["results_plot_filename"] != ""
do_final_model_plot = params["final_model_plot_filename"] != ""

if evaluation:
    results_eval_filename = add_path(write_dir, params["results_eval_filename"])
    results_fx_filename = add_path(write_dir, params["results_fx_filename"])
    results_eval_plot_filename = add_path(write_dir, params["results_eval_plot_filename"])
    do_results_eval_plot = params["results_eval_plot_filename"] != ""

# Get "true" value by using very small time step
#if final_tols is not None:
# y_accurate = run_chem_equil_model(final_tols[0], total_time, plot_filename = "", evaluation = evaluation)
y_accurate = np.pow(total_time + 1, 3/2)

# Values to try
X = np.array(X)

# Apply SPRE
# Define options
# "name"   : str, one of {"MRE", "GRE", "SPRE"} (default: "SPRE")
# "k_name" : str, one of {"Gaussian", "GaussianARD", "Matern1/2", "Matern3/2", "white"} (default: "white")
options = {
    "name": params["extrapolation_name"],
    "k_name":  params["extrapolation_kernel"], 
    "plot" : False
}

# Loop thro' different scalar values for multiplying set of tolerences
for i, h in enumerate(h_values):
    # Results, Y is model output
    Y = np.array([])
    extrapolation_results = []

    # Get results
    for x in X:
        if evaluation:
            y = run_chem_equil_model(h * x, total_time, evaluation = evaluation)
        else:
            y = run_chem_equil_model(h * x[0], total_time)

        Y = np.append(Y, y)

    #print(f"X = {X}")
    #print(f"Y = {Y}")
 
    # Assume extrapolation is a defined function returning a dict with 'mu' and 'var'
    if do_results_plot:
        options["plot_filename"] = results_plot_filename.replace(".png", f"_LOOCV_{i}.png").replace("chem_equil\\", "chem_equil\\loocv_plots\\")

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
    if evaluation:
        # Create row of results for absolute error table
        # h, true_value, best_estimate, spre_estimate, abs_err_best_estimate, abs_err_spre_estimate
        # Also record results for f(0) for all values of x in X, given in Y
        table_row = np.array([h, y_accurate, Y[0], out['mu'][0], np.abs(y_accurate - Y[0]), np.abs(y_accurate - out['mu'][0])])

        if i == 0:
            abs_error_table = np.matrix(table_row)
            fx_table = np.matrix(Y)
        else:
            abs_error_table = np.vstack((abs_error_table, table_row))
            fx_table = np.vstack((fx_table, Y))

# Create dataframe of results
number_of_x = X.shape[0]
header = ["h", "mu", "var"] + [f"mu_cv{n}" for n in range(1, number_of_x + 1)] + [f"var_cv{n}" for n in range(1, number_of_x + 1)]

# Create DataFrame
df = pd.DataFrame(all_extrapolation_results, columns=header)

# Write results to file
if results_filename:
    # Write to file with tab separation
    df.to_csv(results_filename, sep="\t", index=False)

if evaluation:
    # Create table of absolute errors (wrt to "true" value) with
    # 1) Smallest time step in set of time steps
    # 2) SPRE estimate using all time steps in set

    #abs_error_table = np.vstack((all_extrapolation_results, extrapolation_results))

    # Create DataFrame
    abs_header = ["h", "true_value", "best_estimate", "spre_estimate", "abs_err_best_estimate", "abs_err_spre_estimate"]
    df_abs = pd.DataFrame(abs_error_table, columns=abs_header)

    # Write results to file
    if results_eval_filename:
        # Write to file with tab separation
        df_abs.to_csv(results_eval_filename, sep="\t", index=False)

    if do_results_eval_plot:
        # Plot absolute errors of two estimates

        plt.close('all') 
        plt.figure()
        plt.plot(df_abs["h"], df_abs["abs_err_best_estimate"], marker='o', linestyle='solid', linewidth=2, markersize=12, label="best estimate")
        plt.plot(df_abs["h"], df_abs["abs_err_spre_estimate"], marker='o', linestyle='solid', linewidth=2, markersize=12, label="SPRE estimate")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("time step size")
        plt.ylabel("absolute error")
        plt.title("Absolute Errors of f(0) Estimates")
        plt.grid(True)
        plt.legend()
        plt.savefig(results_eval_plot_filename) 
        plt.show()

    # Write f(0) results to file
    if results_fx_filename:
        # Write to file with tab separation
        df_fx = pd.DataFrame(fx_table)
        df_fx.to_csv(results_fx_filename, sep="\t", index=False, columns=None)

else:
    if do_results_plot:
        # Calculate error bars (standard deviation = sqrt(var))
        errors = np.sqrt(df["var"])

        # Plot with error bars
        plt.close('all') 
        plt.figure()
        plt.errorbar(df["h"], df["mu"], yerr=errors, fmt='o-', capsize=5, ecolor='black', markersize=6)
        plt.xlabel("h")
        plt.ylabel("mu")
        plt.title("Extrapolation Results")
        plt.grid(True)

        # Add horizontal dashed line with computed accurate answer
        if final_tols is not None: 
            
            print(f"\nAccurate prediction using step {final_tols[0]} and integral tolerance {final_tols[1]} gives f(0) = {y_accurate}\n")

            #diff = abs(out['mu'][0] - y_accuarte)
            #print(f"This is a difference of {diff:.4f}")
            plt.axhline(y = y_accurate, color='red', linestyle='--', linewidth=1)
    
        plt.savefig(results_plot_filename) 
        plt.show()


if do_final_model_plot:
    _ = run_chem_equil_model(final_tols[0], total_time, plot_filename = final_model_plot_filename)




