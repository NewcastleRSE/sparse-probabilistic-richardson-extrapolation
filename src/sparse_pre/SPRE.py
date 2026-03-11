##############################################################################
# Sparse Probabilistic Richardson Extrapolation (SPRE)
# Main class with methods for peforming SPRE. See test.py and extrapolation.py
# for an example application. 
# Based on the methods and original MatLab code by Chris Oates.
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import jax.numpy as jnp
from jax import grad, debug, hessian, jit, lax
from tqdm import tqdm  # For progress bars
import numpy as np
from scipy.optimize import minimize
from typing import Any

# Ensure 64-bit accuracy is used
from jax import config
config.update("jax_enable_x64", True)

# Application modules
from sparse_pre.helper_functions import x2fx, softplus, cellsum, white, remove_row, stepwise

class SPRE:
    """
    Sparse Probabilistic Richardson Extrapolation (SPRE).
    """

    def __init__(self, kernel_spec : str, dimension : int, gre_base : jnp.ndarray = None):
        """
        Sets up the SPRE class with an initial kernel for extrapolation.

        Parameters:  
            kernel_spec : str               Name of the kernel to set up and use.
            dimension : int                 Data dimension.
            gre_base : jnp.ndarray          Basis for compatability layer for GRE.          
        Returns:
            None         
        """
        
        # Smallest allowed amplitude, epsilson, used in the kernel.
        self.ep = 1e-16  

        # Set dimension, the number of parameters that are discretized in the simulation, such as time
        self.dimension = dimension

        # Set up the kernel to use
        self.set_kernel_spec(kernel_spec, gre_base)


    def cdist_jax(self, XA : jnp.ndarray, XB : jnp.ndarray, squared : bool = False) -> jnp.ndarray:
        """
        Computes pairwise Euclidean distances between two sets of vectors (rows of XA and XB).
        Equivalent to scipy.spatial.distance.cdist(XA, XB, 'euclidean') but using jax.

        Parameters:  
            XA : jnp.ndarray          First array,  (m, d)        
            XB : jnp.ndarray          Second array, (n, d)   
            squared : bool            Return squared distance
        Returns:
            jnp.ndarray               Distances between rows in XA and XB, (m, n)
        """
    
        # ||a - b||² = ||a||² + ||b||² - 2a·b
        XA_sq = jnp.sum(XA ** 2, axis = 1, keepdims = True)  # (m, 1)
        XB_sq = jnp.sum(XB ** 2, axis = 1)  # (n,)
        cross_term = jnp.dot(XA, XB.T)  # (m, n)
       
        # Ensure no problems with negative sqrt if value is -1e16
        nums = XA_sq - 2 * cross_term + XB_sq
        if squared:
            # Return distance squared
            dists = jnp.where(nums >= 0, nums, 0.0)
        else:
            dists = jnp.where(nums >= 0, jnp.sqrt(nums), 0.0)
        
        return dists


    def set_kernel_spec(self, kernel_spec : str, gre_base : jnp.ndarray = None) -> None:
        """
        Sets up the kernel function for analyses, such that kernal(X1, X2, x) returns the set kernel function
        where X1 and X2 are simulation data output and x is array of hyperparameters for the kernel.
            X1 = n1 x dimension
            X2 = n2 x dimension
            x = p x 1
        Where n1 and n2 are the number of observations and p the number of kernel hyperparameters.

        Parameters:  
            kernel_spec : str               Name of the kernel to set up and use         
            gre_base : jnp.ndarray          Basis for compatability layer for GRE             
        Returns:
            None         
        """

        if gre_base is None:
            # Create kernal function
            self.kernel_spec = kernel_spec
            self.kernel_base = None
            
        else: 
            # Compatability layer for GRE 
            self.kernel_spec = "GRE"
            self.kernel_base = kernel_spec
              
        # Set GRE base (set to None if not used)
        self.gre_base = gre_base

        # Set default parameters
        self.set_kernel_default_parameters() 
       

    def set_kernel_default_parameters(self) -> None:
        """
        Sets up the default hyperparameters for the set kernel function.
      
        Parameters:  
            None               
        Returns:
            None         
        """
       
        # Set default parameters
        match self.kernel_spec:
            case "Gaussian":
                self.default_kernel_parameters = [1.0, 0.1]                 
            case "GaussianARD":
                self.default_kernel_parameters = [1.0] 
                self.default_kernel_parameters.extend([0.1] * self.dimension)      
            case "white":
                self.default_kernel_parameters = [1.0] 
            case "Matern1/2":
                self.default_kernel_parameters = [1.0, 1.0]   
            case "Matern3/2":
                self.default_kernel_parameters = [1.0, 1.0]
            case "GRE":
                # Get default kernel parameters for kernel base
                self.kernel_spec = self.kernel_base
                self.set_kernel_default_parameters()
                # Set base and spec variable back
                self.kernel_base = self.kernel_spec
                self.kernel_spec = "GRE"
                # Set final default kernel parameters               
                base_default_parameters = self.default_kernel_parameters
                self.default_kernel_parameters = [1.0] 
                self.default_kernel_parameters.extend(base_default_parameters)                  
            case _:
                raise ValueError(f"Unknown kernel specification: {self.kernel_spec}")
            

    def kernel(self, X1 : jnp.ndarray, X2 : jnp.ndarray, x  : jnp.ndarray = None) -> jnp.ndarray:
        """
        Returns evalution of the kernel function. 
      
        Parameters:  
            X1 : jnp.ndarray          First array    
            X2 : jnp.ndarray          Second array
            x : jnp.ndarray           Vector of kernel hyperparameters to use when evaluating the kernel
                 
        Returns:
            jnp.ndarray                
        """

        # Use default kernel parameters if not set
        if x is None:
            x = self.default_kernel_parameters

        # Calculate the kernel depending on the set kernel to use
        match self.kernel_spec:
            case "Gaussian":                
                return (self.ep + softplus(x[0])) * jnp.exp((-self.cdist_jax(X1, X2) ** 2)/ softplus(x[1])**2)
            
            case "GaussianARD":
                x = jnp.asarray(x)

                amp = self.ep + softplus(x[0])

                lengthscales = softplus(x[1:self.dimension + 1])

                X1_scaled = X1 / lengthscales
                X2_scaled = X2 / lengthscales

                r2 = self.cdist_jax(X1_scaled, X2_scaled, squared=True)

                return amp * jnp.exp(-r2)
            
            case "white":
                return (self.ep + softplus(x[0])) * white(X1, X2)

            case "Matern1/2":
                return (self.ep + softplus(x[0])) * jnp.exp(-self.cdist_jax(X1, X2) / softplus(x[1]))

            case "Matern3/2":              
                l = softplus(x[1])
                sqrt3_r_l = jnp.sqrt(3) * self.cdist_jax(X1, X2) / l
                return (self.ep + softplus(x[0])) * (1 + sqrt3_r_l) * jnp.exp(-sqrt3_r_l)
            
            case "GRE":
                amp = self.ep + softplus(x[0])
                # Convergence rate ansatz b(x)
                base_X1 = jnp.sum(x2fx(X1, self.gre_base), axis=1)
                base_X2 = jnp.sum(x2fx(X2, self.gre_base), axis=1)
                self.kernel_spec = self.kernel_base
                ans = amp * base_X1[:, None] * self.kernel(X1, X2, x = x[1:]) * base_X2[None, :]
                self.kernel_spec = "GRE"
                return ans 


    def cv_local_loss(self, x : jnp.ndarray, A : jnp.ndarray, row_num : int, return_mu_cov : bool = False) -> Any:
        """
        Sets arrays to use for cross-validation local loss (log-likelihood of test data) and returns result.
      
        Parameters:    
            x : jnp.ndarray           Vector of kernel hyperparameters to use when evaluating the kernel
            A : jnp.ndarray           binary matrix representing the sparse basis  
            row_num : int             Row number to leave out for leave-one-out cross validation.
            return_mu_cov : bool      Whether to return mu and cov instead of the local loss    
        Returns:
            float or tuple               
        """

        # X = n_train x d
        # Y = n_train x 1
        # Xs = n_test x d
        # Ys = n_test x 1
        X = remove_row(self.X_normalised, row_num)
        Y = remove_row(self.Y_normalised, row_num)
        Xs = self.X_normalised[row_num:(row_num+1), :]
        Ys = self.Y_normalised[row_num:(row_num+1)]

        return self.cv_loss_calculation(A, X, Y, Xs, Ys, x, return_mu_cov)


    def check_unisolvent(self, A : jnp.ndarray) -> int:
        """
        Checks whether the basis A produces a unisolvent set.
        Returns an int (rather than bool) in order to allow use of JAX JIT (just in time compilation).
      
        Parameters:  
            A : jnp.ndarray           binary matrix representing the sparse basis        
        Returns:
            int              
        """

        m = A.shape[0]
        VA = x2fx(self.X_normalised, A)
        rank = jnp.linalg.matrix_rank(VA)

        def on_true(_):
            # everything OK
            return 1

        def on_false(_):    
            # Raising Python errors inside JIT is not allowed.
            # Instead return a special value.
            debug.print("\nWARNING: A non-unisolvent set encountered! Rank={rank}, m={m}", rank=rank, m=m)       
            return -1   

        return lax.cond(rank == m, on_true, on_false, operand = None)


    def cv_loss_calculation(self, A : jnp.ndarray, X : jnp.ndarray, Y : jnp.ndarray, Xs : jnp.ndarray, Ys : jnp.ndarray, x : jnp.ndarray, return_mu_cov : bool = False) -> Any:
        """
        Calculates cross-validation local loss (log-likelihood of test data).
      
        Parameters:  
            A : jnp.ndarray           binary matrix representing the sparse basis     
            X : jnp.ndarray           Input array of discretized parameters for simulation model 
            Y : jnp.ndarray           Output values of the simulation model
            Xs : jnp.ndarray          The left-out row of X  
            Ys : jnp.ndarray          The left-out row of Y
            x : jnp.ndarray           Vector of kernel hyperparameters to use when evaluating the kernel
            row_num_str : str         Row number to leave out for leave-one-out cross validation.
            return_mu_cov : bool      Whether to return mu and cov instead of the loss    
        Returns:
            float or tuple               
        """

        # Cross-validation local loss (log-likelihood of test data)
        # A = m x d (sparse matrix)
        # X = n_train x d
        # Y = n_train x 1
        # Xs = n_test x d
        # Ys = n_test x 1
        # x = p x 1
    
        # Calculate some bits firstly    
        K_inv = jnp.linalg.inv(self.kernel(X, X, x))
        kernel_Xs_Xs = self.kernel(Xs, Xs, x)
        kernel_X_Xs = self.kernel(X, Xs, x)
        
        # For most kernels kernel_Xs_X and kernel_X_Xs are the same
        if self.kernel_spec != "GRE":            
            kernel_Xs_X = kernel_X_Xs.T 
        else:
            kernel_Xs_X = self.kernel(Xs, X, x)

        # Basis functions
        # A = m x d (sparse matrix)
        # X = n x d
        # Xs = n_test x d
        VA = x2fx(X, A)
        vAT = x2fx(Xs, A).T
         
        # Residual term
        # A = m x d (sparse matrix)
        # X = n_train x d
        # Xs = n_test x d
        # x = p x 1   
        VA_T_at_K_inv = VA.T @ K_inv  
        residual_X_Xs = vAT - VA_T_at_K_inv @ kernel_X_Xs
    
        # Predictive covariance
        # A = m x d (sparse matrix)
        # X = n_train x d
        # Xs = n_test x d
        # x = p x 1  
        inv_VA_T_at_K_inv_at_VA = jnp.linalg.inv(VA_T_at_K_inv @ VA)
        cov_val = (kernel_Xs_Xs
                - kernel_Xs_X @ K_inv @ kernel_X_Xs
                + residual_X_Xs.T @ inv_VA_T_at_K_inv_at_VA @ residual_X_Xs)
        
        # Coefficient estimator, beta
        # A = m x d (sparse matrix)
        # X = n_train x d
        # Y = n_train x 1
        # x = p x 1          
        beta_X_Y = inv_VA_T_at_K_inv_at_VA @ (VA_T_at_K_inv @ Y)
        
        # Predictive mean
        # A = m x d (sparse matrix)
        # X = n_train x d
        # Y = n_train x 1
        # Xs = n_test x d
        # x = p x 1
        mu_val = kernel_Xs_X @ K_inv @ Y + residual_X_Xs.T @ beta_X_Y

        # Return mu and cov instead
        if return_mu_cov:
            # Avoid numerical error giving negative values
            if cov_val[0][0] < 0:
                cov_val = cov_val.at[0].set(0)
            return mu_val, cov_val

        diff = Ys - mu_val
        inv_cov = jnp.linalg.inv(cov_val)
        term1 = -0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * cov_val))
        term2 = -0.5 * diff.T @ inv_cov @ diff
        
        return term1 + term2


    def cv_loss(self, x : jnp.ndarray, A : jnp.ndarray) -> float:
        """
        Calculate the loss (log-likelihood of test data) using leave-one-out cross validation (LOOCV).
      
        Parameters:  
            x : jnp.ndarray           Vector of kernel hyperparameters to use when evaluating the kernel
            A : jnp.ndarray           binary matrix representing the sparse basis 
        Returns:
            float              
        """

        # A = m x d (sparse matrix)
        # X = n_train x d
        # Y = n_train x 1
        # x = p x 1
        return sum(
            self.cv_local_loss(x, A, i) for i in range(self.X_normalised.shape[0])
        )


    def set_normalised_data(self, X : jnp.ndarray, Y : jnp.ndarray) -> None:
        """
        Set normalised data and save as object variables for use in other methods.

        Parameters:
            A : jnp.ndarray             shape (m, d), binary matrix representing the sparse basis
            X : jnp.ndarray             shape (n_train, d), training inputs
            Y : jnp.ndarray             shape (n_train,), training outputs

        Returns:
            None
        """
          
        # Data normalization   
        self.nX = self.ep + (jnp.max(X, axis=0) - jnp.min(X, axis=0))
        self.nY = self.ep + (jnp.max(Y) - jnp.min(Y))
        self.X_normalised = X / self.nX
        self.Y_normalised = Y / self.nY
    

    def perform_extrapolation(self, x : jnp.ndarray, A : jnp.ndarray, return_mu_and_var : bool = False) -> dict:
        """
        Perform Sparse Probabilistic Richardson Extrapolation (SPRE) for the given values of
        the kernel parameters.
        
        Parameters:
            x : jnp.ndarray             shape (p,), kernel parameters
            A : jnp.ndarray             binary matrix representing the sparse basis
            return_mu_and_var : bool    Whether to return variables: mu, mu_cv, var and var_cv

        Returns:
            out : dict with keys:
                - mu: scalar, predictive mean for f(0)
                - var: scalar, predictive variance for f(0)
                - mu_cv: n_train x 1, leave-one-out cross validation (LOOCV) predictive means
                - var_cv: n_train x 1, LOOCV predictive variances
                - cv: scalar, LOOCV criterion
                - cv_grad: p x 1, gradient of LOOCV criterion
        """
    
        # Evaluate cv
        cv = self.cv_loss(x, A)

        # Output
        out = {
            "cv": cv       
        }

        # Uncomment to output info on fitting kernel parameters
        # As table easy to copy and paste with neg log like and gradient    
        #gradient_function = grad(self.cv_loss) # Define the gradient function 
        #gradient = gradient_function(x) # Evaluate gradient at x
        #debug.print("{}, {}, {}, {}, {}", -cv, -gradient[0], -gradient[1], x[0], x[1]) 

        # Add extra output if requested
        if return_mu_and_var:
            # LOOCV predictions
            mu_cv = jnp.zeros(self.X_normalised.shape[0])
            var_cv = jnp.zeros(self.X_normalised.shape[0])
            for i in range(self.X_normalised.shape[0]):               
                mu_val, cov_val = self.cv_local_loss(x, A, i, return_mu_cov = True)               
                mu_cv = mu_cv.at[i].set((self.nY * mu_val[0]))              
                var_cv = var_cv.at[i].set((self.nY**2 * cov_val[0][0]))
              
            # Output
            mu_value, cov_value = self.cv_loss_calculation(A, self.X_normalised, self.Y_normalised, jnp.zeros((1, self.dimension)), jnp.zeros((1, self.dimension)), x, return_mu_cov = True)
            out0 = out
            out = {
                "mu": self.nY * mu_value,
                "var": self.nY**2 * cov_value,           
                "mu_cv": mu_cv,
                "var_cv": var_cv            
            }
            out.update(out0)
            
        return out


    def objective(self, x : jnp.ndarray, A : jnp.ndarray) -> float:
        """
        Objective function used to fit the hyperparameters of the kernel.
        
        Parameters:
            x : jnp.ndarray             shape (p,), kernel hyperparameters
            A : jnp.ndarray             binary matrix representing the sparse basis
        Returns:
            float
        """

        # Return the negative log likelihood using LOOCV with gradient
        out = self.jit_perform_extrapolation(x, A)     
        return -out['cv'] 


    def scipy_hess(self, x_np : np.ndarray, A : np.ndarray) -> np.ndarray:
        """
        Wrapper to create hessian using numpy arrays using JAX calculated hessian.

        Parameters:
            x_np : np.ndarray         shape (p,), kernel hyperparameters
            A : np.ndarray            binary matrix representing the sparse basis
        Returns:
            np.ndarray
        """

        x_jnp = jnp.asarray(x_np)
        A_jnp = jnp.asarray(A)
        # Negate for negative log likelihood
        return -np.asarray(self.jit_hess(x_jnp, A_jnp))


    def scipy_objective(self, x_np : np.ndarray, A : np.ndarray) -> np.float64:
        """
        Wrapper to create objective function using numpy arrays.

        Parameters:
            x_np : np.ndarray         shape (p,), kernel hyperparameters
            A : np.ndarray            binary matrix representing the sparse basis
        Returns:
            np.float64
        """

        x_jnp = jnp.asarray(x_np)            # numpy -> jax
        A_jnp = jnp.asarray(A)
        return self.objective(x_jnp, A_jnp).astype(np.float64)


    def scipy_jac(self, x_np : np.ndarray, A : np.ndarray) -> np.ndarray:
        """
        Wrapper to create Jacobian (gradient) using numpy arrays using JAX calculated Jacobian.

        Parameters:
            x_np : np.ndarray         shape (p,), kernel hyperparameters
            A : np.ndarray            binary matrix representing the sparse basis
        Returns:
            np.ndarray
        """

        x_jnp = jnp.asarray(x_np)
        A_jnp = jnp.asarray(A)
        # Negate for negative log likelihood
        return -np.asarray(self.jit_grad(x_jnp, A_jnp))     # return numpy array


    def perform_extrapolation_optimization(self, A : jnp.ndarray, do_jit : bool = True) -> dict:
        """
        Optimize kernel hyperparameters for SPRE.

        Parameters:
            A : jnp.ndarray           binary matrix representing the sparse basis
            do_jit : bool             Do "Just in time" compilation to speed up the fitting.
                                      all "self" values must remain constant during this call otherwise
                                      this should be set to True.
        Returns:
            out : dict with keys:
                 x  = p x 1, fitted kernel hyperparameters 
                 cv = scalar, LOOCV criterion
        """

        if do_jit:
            self.prepare_jit_for_extrapolation_optimization()

        result = minimize(self.scipy_objective,
                    self.default_kernel_parameters,                    
                    method='trust-krylov',   # trust-krylov is trust region fitting algorithm
                    jac=self.scipy_jac,      # gradient
                    hess=self.scipy_hess,    # hessian
                    args=(A),
                    options={'maxiter': 1000, 'disp': False})

        result_value = self.objective(result.x, A)
        result_params = result.x
       
        return {
            'x'  : result_params,
            'cv' : result_value
        }


    def prepare_jit_for_extrapolation_optimization(self) -> None:
        """
        Create gradient, hessian and extrapolation functions using JAX "Just in time" (JIT)
        compilation to speed up calculations when fitting parameters.

        Parameters:
            None
        Returns:
            None
        """

        # "Just in time" compilation to speed up the fitting.
        # Note: object variables are not allowed to change when these functions are called.
        self.jit_hess = jit(hessian(self.cv_loss))
        self.jit_grad = jit(grad(self.cv_loss))
        self.jit_perform_extrapolation = jit(self.perform_extrapolation)


    def stepwise_selection(self, max_order : int = 0, use_fixed_basis : bool = False, bases_filename : str = "", h : float = None) -> dict:
        """
        Stepwise model selection for SPRE.

        Parameters:
            max_order : int             maximum order number to fit. Zero sets no limit.
                                        (to avoid never ending orders being used for problematic datasets)
            use_fixed_basis : bool      whether to use fixed default basis for SPRE,
                                        e.g. for d=2 use A=[[0, 0], [1, 0], [0, 1]]
            bases_filename : str        optional filename to record the bases in
            h : float                   float for h value to record in bases file

        Returns:
            out     : dict, result of SPRE using optimal model
                    out.mu      = scalar, predictive mean for f(0)
                    out.var     = scalar, predictive variance for f(0)
                    out.mu_GP   = function R^d -> R, predictive mean for fitted GP
                    out.cov_GP  = function R^d x R^d -> R, predictive covariance for fitted GP
                    out.mu_cv   = n_train x 1, LOOCV predictive means
                    out.var_cv  = n_train x 1, LOOCV predictive variances
                    out.cv      = scalar, LOOCV criterion
                    out.cv_grad = p x 1, gradient of LOOCV criterion
        """

        # Number of training points
        n_train = self.X_normalised.shape[0]

        # Initialise with just an intercept
        A = jnp.zeros((1, self.dimension), dtype=int)  

        # Handle selection differently if doing GRE
        if self.kernel_base is not None:
            return self._GRE_stepwise_selection(A, max_order)
        
        # Do "Just In Time" JIT compilation to speed up the fitting.
        # Required before running perform_extrapolation_optimization
        self.prepare_jit_for_extrapolation_optimization()

        # No need to do JIT again
        do_jit = False

        # Try higher orders
        carry_on = True
        
        if use_fixed_basis:
            # Fix basis to default basis and keep fixed.
            A = jnp.vstack((A, jnp.eye(self.dimension, dtype=int)))
            carry_on = False
        
        order = 0
        fit = self.perform_extrapolation_optimization(A, do_jit)
        cv = fit['cv']
        
        # Try expanding basis A for a better fit
        while carry_on and (max_order == 0 or order < max_order):            
            m = A.shape[0] # Number of rows in base A
            order += 1 # Consider the addition of higher order interactions
            A_extra = stepwise(A, order)  # All predictors of the next order to consider
            n_extra = A_extra.shape[0]
            to_include = jnp.zeros(n_extra, dtype=bool)

            print(f"Fitting interactions of order {order}:")

            for i in tqdm(range(n_extra), desc="Stepwise progress"):
                A_new = jnp.vstack([A, A_extra[i]])  
                ## Check maximum rank, must be m to be OK, m = number of rows in A               
                if self.check_unisolvent(A_new) > 0:             
                    fit_new = self.perform_extrapolation_optimization(A_new, do_jit)
                    cv_new = fit_new['cv']
                    if cv_new < cv: # If adding new predictor helped
                        to_include = to_include.at[i].set(True)

            if jnp.any(to_include) and ((m + sum(to_include)) < (n_train - 1)):
                A_updated = jnp.vstack([A, A_extra[to_include]])                          
                fit_updated = self.perform_extrapolation_optimization(A_updated, do_jit)
                cv_updated = fit_updated['cv']
                if cv_updated >= cv:             
                    carry_on = False
                else:
                    A = A_updated
                    fit = fit_updated
                    cv = cv_updated
            else:
                carry_on = False

        # Optimal parameters
        x_opt = fit['x']
       
        # Final model with best kernel parameters and basis A
        out = self.perform_extrapolation(x_opt, A, return_mu_and_var = True)

        # Record used basis
        if bases_filename != "":
            with open(bases_filename, "a") as f:
                if h is not None:
                    f.write(f"\nh = {h}, Basis Matrix:\n")
                else:
                    f.write("\nBasis Matrix:\n")
                np.savetxt(f, A, fmt='%d')              

        return out
    

    def _GRE_stepwise_selection(self, A : jnp.ndarray, max_order : int = 0) -> dict:
        """
        Stepwise model selection for GRE (Gauss-Richardson Extrapolation).

        Parameters:
            A : jnp.ndarray           binary matrix representing the sparse basis
            max_order : int           maximum order number to fit. Zero sets no limit.
                                        (to avoid never ending orders being used for problematic datasets)
        Returns:
            out : dict
                Dictionary with predictive mean, variance, and fitted model details.
        """
       
        # Do "Just In Time" JIT compilation to speed up the fitting.
        # Required before running perform_extrapolation_optimization
        self.prepare_jit_for_extrapolation_optimization()

        # Need to update code to pass B to perform_extrapolation_optimization
        #  if this can be set to False
        do_jit = True

        # initial rate function: only intercept
        B = jnp.zeros((1, self.dimension), dtype=int)      
        order = 0

        fit = self.perform_extrapolation_optimization(A, do_jit)
        cv = fit["cv"]

        carry_on = True
        while carry_on and (max_order == 0 or order < max_order):
            order += 1
            B_extra = stepwise(B, order)  # Generate all predictors of the next order
            n_extra = B_extra.shape[0]

            print(f"Fitting interactions of order {order}...")

            to_include = jnp.zeros(n_extra, dtype=bool)
            for i in tqdm(range(n_extra), desc="Stepwise progress"):
                B_new = jnp.vstack([B, B_extra[i, :]])
                self.set_kernel_spec(self.kernel_base, B_new)               
                fit_new = self.perform_extrapolation_optimization(A, do_jit)
                cv_new = fit_new["cv"]

                if cv_new < cv:                    
                    to_include = to_include.at[i].set(True)

            if jnp.any(to_include):
                B_updated = jnp.vstack([B, B_extra[to_include, :]])
                self.set_kernel_spec(self.kernel_base, B_updated)             
                fit_updated = self.perform_extrapolation_optimization(A, do_jit)
                cv_updated = fit_updated["cv"]

                if cv_updated >= cv:
                    carry_on = False                    
                else:
                    fit = fit_updated
                    B = B_updated
                    cv = cv_updated
            else:
                carry_on = False

        # Final GP fit with optimal parameters
        x_opt = fit["x"]       
        self.set_kernel_spec(self.kernel_base, B)   
        self.prepare_jit_for_extrapolation_optimization()  
        out = self.perform_extrapolation(x_opt, A, return_mu_and_var=True)

        return out
    