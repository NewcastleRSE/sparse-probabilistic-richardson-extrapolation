##############################################################################
# SIR model used as an example for the SPRE method
# Susceptible, Infectious (or Infected) and Recovered (or Removed)
#
# From root directory, for example run
# python .\src\sir.py .\data\sir\input_1.json
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

def sir_model(t : float, y : tuple, beta : float, gamma : float, N : int) -> tuple:
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
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return (dSdt, dIdt, dRdt)

def run_sir_model(diff_tol : float = 1e-8, integrate_tol : float = 1e-8,
                   N : int = 1000, beta: float = 0.3, gamma : float = 0.1,
                   initial_infected : int = 1, total_time : float = 120, plot_filename : str = ""):
    """
    Runs SIR model and saves results    
    """
   
    
    S0 = N - initial_infected
    I0 = initial_infected
    R0 = 0
    y0 = [S0, I0, R0]

    # Time span to evalute the SIR model
    t_span = (0, total_time)

    # Timepoints at which to store values
    number_of_points = 100000 #max(2, int(1/diff_tol)) + 1
    t_eval = np.linspace(*t_span, number_of_points)

    # -----------------------------
    # Solve SIR system together
    # -----------------------------
    sol = solve_ivp(
        fun=sir_model,
        t_span=t_span,
        y0=y0,
        args=(beta, gamma, N),
        t_eval=t_eval,
        method='RK45',
        rtol=diff_tol  
    )

    S, I, R = sol.y
    t = sol.t

    if plot_filename:
        # -----------------------------
        # Plot Results
        # -----------------------------
        plt.close('all') 
        plt.figure(figsize=(10, 6))
        plt.plot(t, S, label='Susceptible')
        plt.plot(t, I, label='Infected')
        plt.plot(t, R, label='Recovered')
        plt.xlabel('Time (days)')
        plt.ylabel('Population')
        plt.title('SIR Model')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(plot_filename)  
        plt.show()

    # -----------------------------
    # Tolerance-controlled integral of I(t)
    # -----------------------------
    # Interpolate I(t) for smooth integration
    I_interp = interp1d(t, I, kind='cubic', fill_value="extrapolate")

    # Integrate I(t) using adaptive quadrature (quad)
    infected_person_days, err = quad(I_interp, t_span[0], t_span[1], epsrel=integrate_tol)

    print(f"Estimate using R(120)/gamma is {R[-1]/gamma}\n")

    # Return the total number of "person-days of infection"
    # Indicates the burden on a helathcare system
    return infected_person_days

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

# Total indviduals
N = params["N"]

# Infection rate
beta = params["beta"]

# Recovery rate
gamma = params["gamma"]

initial_infected = params["initial_infected"]
total_time = params["total_time"]
X = params["X"]
h_values = params["h_values"]
final_tols = params["final_tols"]

# Files to save results
def add_path(path, filename):
    new_filename = ""
    if filename is not None and filename:
        new_filename = os.path.join(path, filename)
    return new_filename

results_filename = add_path(write_dir, params["results_filename"])
results_plot_filename = add_path(write_dir, params["results_plot_filename"])
final_sir_plot_filename = add_path(write_dir, params["final_sir_plot_filename"])

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
        y = run_sir_model(h * x[0], h * x[1], N, beta, gamma, initial_infected, total_time)
        Y = np.append(Y, y)

    #print(f"X = {X}")
    #print(f"Y = {Y}")
 
    # Assume extrapolation is a defined function returning a dict with 'mu' and 'var'
    if results_plot_filename:
        options["plot_filename"] = results_plot_filename.replace(".png", f"_LOOCV_{i}.png").replace("sir\\", "sir\\loocv_plots\\")

    out = extrapolation(X, Y, options)
    print(f"Predict f(0) = {out['mu'][0]} +/- {np.sqrt(out['var'][0][0])}\n")
    
    extrapolation_results.extend([h, out['mu'][0], out['var'][0][0]])

    # Append results for each point
    extrapolation_results.extend(out['mu_cv'])
    extrapolation_results.extend(out['var_cv'])
  
    if i == 0:
        all_extrapolation_results = extrapolation_results
    else:
        all_extrapolation_results = np.vstack((all_extrapolation_results, extrapolation_results))


# Create dataframe of results
number_of_x = X.shape[0]
header = ["h", "mu", "var"] + [f"mu_cv{n}" for n in range(1, number_of_x + 1)] + [f"var_cv{n}" for n in range(1, number_of_x + 1)]

# Create DataFrame
df = pd.DataFrame(all_extrapolation_results, columns=header)

# Write results to file
if results_filename:
    # Write to file with tab separation
    df.to_csv(results_filename, sep="\t", index=False)

if results_plot_filename:
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
        
        y_accuarte = run_sir_model(final_tols[0], final_tols[1], N, beta, gamma, initial_infected, total_time, plot_filename = "")

        print(f"\nAccurate prediction using step {final_tols[0]} and integral tolerance {final_tols[1]} gives f(0) = {y_accuarte}\n")

        #diff = abs(out['mu'][0] - y_accuarte)
        #print(f"This is a difference of {diff:.4f}")
        plt.axhline(y = y_accuarte, color='red', linestyle='--', linewidth=1)
  
    plt.savefig(results_plot_filename) 
    plt.show()

    if final_sir_plot_filename:
        y_accuarte = run_sir_model(final_tols[0], final_tols[1], N, beta, gamma, initial_infected, total_time, plot_filename = final_sir_plot_filename)




