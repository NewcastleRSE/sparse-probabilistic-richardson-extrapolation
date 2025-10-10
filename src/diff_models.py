##############################################################################
# Differential equation models
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt

# Application modules
from sparse_pre.extrapolation import extrapolation

class Model:
    """
    Base model class
    """

    def __init__(self):
        # Set default parameters values  
        self.total_time = 120


    def diff_model(self, t : float, y : tuple) -> tuple:
        return (0, 0)
       
    def get_initial_condition(self) -> tuple:
        return (0, 0)
    
    def get_final_quantity(self) -> float:
        return 0
    
    def set_parameters(self, parameters : dict):
        
        """Set variables from a dictionary."""
        for key, value in parameters.items():
            setattr(self, key, value)

    def solve_diff_equations(self, diff_tol : float = 1e-8) -> float:
        """
        Runs model   
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
            rtol=diff_tol  
        )

        self.diff_solution = sol

        return self.get_final_quantity()
       
    def plot_diff_solution(self):

        # -----------------------------
        # Plot Results
        # -----------------------------
        plt.close('all') 
        plt.figure(figsize=(10, 6))
        for y in self.diff_solution.y:
            plt.plot(self.diff_solution.y, y, label='')
           
        plt.xlabel('time')
        plt.ylabel('')
        plt.title('')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if self.plot_filename:
            plt.savefig(self.plot_filename)  
        plt.show()

class SirModel(Model):
    def __init__(self):
        # Call Parent’s constructor
        super().__init__()
        
    def diff_model(self, t : float, y : tuple, beta : float, gamma : float, N : int) -> tuple:
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
        dSdt = -self.beta * S * I / N
        dIdt = self.beta * S * I / N - self.gamma * I
        dRdt = self.gamma * I
        return (dSdt, dIdt, dRdt)

     
    def get_initial_condition(self) -> tuple:

        S0 = self.N - self.initial_infected
        I0 = self.initial_infected
        R0 = 0
        return (S0, I0, R0)

    def get_final_quantity(self) -> float:
        return 



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
