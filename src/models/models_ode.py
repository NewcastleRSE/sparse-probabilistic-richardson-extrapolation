##############################################################################
# Simulation Models using ODEs
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import numpy as np
import numpy.typing as npt
from scipy.integrate import quad
from scipy.interpolate import interp1d

# Application modules
from models.base_model import Model

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
