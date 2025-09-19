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
from jax import grad, debug
from jaxopt import GradientDescent
from tqdm import tqdm  # For progress bars
from sklearn.neighbors import NearestNeighbors
import numpy as np

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
       
        # Set kernel cache - used to speed up calculations
        self.kernel_cache = {}

    def cdist_jax(self, XA : jnp.ndarray, XB : jnp.ndarray) -> jnp.ndarray:
        """
        Computes pairwise Euclidean distances between two sets of vectors (rows of XA and XB).
        Equivalent to scipy.spatial.distance.cdist(XA, XB, 'euclidean') but using jax.

        Parameters:  
            XA : jnp.ndarray          First array,  (m, d)        
            XB : jnp.ndarray          Second array, (n, d)              
        Returns:
            jnp.ndarray               Distances between rows in XA and XB, (m, n)
        """
    
        # ||a - b||² = ||a||² + ||b||² - 2a·b
        XA_sq = jnp.sum(XA ** 2, axis = 1, keepdims = True)  # (m, 1)
        XB_sq = jnp.sum(XB ** 2, axis = 1)  # (n,)
        cross_term = jnp.dot(XA, XB.T)  # (m, n)
       
        dists = jnp.sqrt(XA_sq - 2 * cross_term + XB_sq)
        return dists

    def set_kernel_spec(self, kernel_spec : str, gre_base : jnp.ndarray = None):
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
            
        else: 
            # Compatability layer for GRE 
            self.kernel_spec = "GRE"
            self.kernel_base = kernel_spec
              
        # Set GRE base (set to None if not used)
        self.gre_base = gre_base

        # Set default parameters
        self.set_kernel_default_parameters() 
       
    def set_kernel_default_parameters(self):
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
            
    def set_kernel_cache(self):
        """
        Calculates and stores values in the kernel function cache. This is used to
        to speed up calculations of the loss.
      
        Parameters:  
            None               
        Returns:
            None         
        """

        # Clear the cache
        self.kernel_cache = {}

        # Loop thro' each row of the data and store calculations in the cache
        for i in range(self.X_normalised.shape[0]):
            X = remove_row(self.X_normalised, i)           
            Xs = self.X_normalised[i:(i+1), :]
            self.kernel_cache[f"XX{i}"] = self.calculate_kernel_cached_bit(X, X)         
            self.kernel_cache[f"XXs{i}"] = self.calculate_kernel_cached_bit(X, Xs) 
            self.kernel_cache[f"XsXs{i}"] = self.calculate_kernel_cached_bit(Xs, Xs)        
            self.kernel_cache[f"XsX{i}"] = self.calculate_kernel_cached_bit(Xs, X) 
              
    def calculate_kernel_cached_bit(self, X1 : jnp.ndarray, X2 : jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the appropriate parts of the kernel function which can be cached and reused depending
        on which kernel is being used. 
      
         Parameters:  
            X1 : jnp.ndarray          First array    
            X2 : jnp.ndarray          Second array           
        Returns:
            jnp.ndarray                
        """

        match self.kernel_spec:
            case "Gaussian":
                return -self.cdist_jax(X1, X2) ** 2 
            
            case "GaussianARD":                
                return None
            
            case "white":
                return white(X1, X2)

            case "Matern1/2":
                return -self.cdist_jax(X1, X2)

            case "Matern3/2":
                return jnp.sqrt(3) * self.cdist_jax(X1, X2)
            
            case "GRE":
                self.kernel_spec = self.kernel_base
                ans = self.calculate_kernel_cached_bit(X1, X2)
                self.kernel_spec = "GRE"
                return ans
            case _:
                raise ValueError(f"Unknown kernel specification: {self.kernel_spec}")

    def get_kernel_cached_bit(self, X1 : jnp.ndarray, X2 : jnp.ndarray, cache_key : str) -> jnp.ndarray:
        """
        Returns the appropriate part of the kernel function which has been cached. 
      
        Parameters:  
            X1 : jnp.ndarray          First array    
            X2 : jnp.ndarray          Second array
            cache_key : str           Name of the cached part to return           
        Returns:
            jnp.ndarray                
        """

        # Check the cached part exists and return it, if not then calculate it.
        if cache_key is not None and cache_key in self.kernel_cache:
            return self.kernel_cache[cache_key]   
        else:
            return self.calculate_kernel_cached_bit(X1, X2)
       
    def kernel(self, X1 : jnp.ndarray, X2 : jnp.ndarray, x  : jnp.ndarray = None, cache_key : str = None) -> jnp.ndarray:
        """
        Returns the appropriate part of the kernel function which has been cached. 
      
        Parameters:  
            X1 : jnp.ndarray          First array    
            X2 : jnp.ndarray          Second array
            x : jnp.ndarray           Vector of kernel hyperparameters to use when evaluating the kernel
            cache_key : str           Name of the cached part to use, if any  
                 
        Returns:
            jnp.ndarray                
        """

        # Use default kernel parameters if not set
        if x is None:
            x = self.default_kernel_parameters

        # Calculate the kernel depending on the set kernel to use
        match self.kernel_spec:
            case "Gaussian":
                return (self.ep + softplus(x[0])) * jnp.exp(self.get_kernel_cached_bit(X1, X2, cache_key) / softplus(x[1])**2)
            
            case "GaussianARD":
                amp = self.ep + softplus(x[0])
                lengthscales = [self.cdist_jax(X1[:, [i]], X2[:, [i]])**2 / softplus(x[i+1])**2 for i in range(self.dimension)]
                return amp * jnp.exp(-cellsum(lengthscales))
            
            case "white":
                return (self.ep + softplus(x[0])) * self.get_kernel_cached_bit(X1, X2, cache_key)

            case "Matern1/2":
                return (self.ep + softplus(x[0])) * jnp.exp(-self.cdist_jax(X1, X2) / softplus(x[1]))

            case "Matern3/2":              
                l = softplus(x[1])
                sqrt3_r_l = self.get_kernel_cached_bit(X1, X2, cache_key) / l
                return (self.ep + softplus(x[0])) * (1 + sqrt3_r_l) * jnp.exp(-sqrt3_r_l)
            
            case "GRE":
                amp = self.ep + softplus(x[0])
                # Convergence rate ansatz b(x)
                base_X1 = jnp.sum(x2fx(X1, self.gre_base), axis=1)
                base_X2 = jnp.sum(x2fx(X2, self.gre_base), axis=1)
                self.kernel_spec = self.kernel_base
                ans = amp * base_X1[:, None] * self.kernel(X1, X2, x = x[1:], cache_key = cache_key) * base_X2[None, :]
                self.kernel_spec = "GRE"
                return ans 

    def cv_local_loss(self, x  : jnp.ndarray, row_num : int, return_mu_cov : bool = False) -> object:
        """
        Sets arrays to use for cross-validation local loss (log-likelihood of test data) and returns result.
      
        Parameters:  
            x : jnp.ndarray           Vector of kernel hyperparameters to use when evaluating the kernel
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

        return self.cv_loss_calculation(X, Y, Xs, Ys, x, str(row_num), return_mu_cov)

    # Loss (log-likelihood of test data)
    def cv_loss_calculation(self, X : jnp.ndarray, Y : jnp.ndarray, Xs : jnp.ndarray, Ys : jnp.ndarray, x : jnp.ndarray, row_num_str : str = "_", return_mu_cov : bool = False):
        """
        Calculates cross-validation local loss (log-likelihood of test data).
      
        Parameters:  
            X : jnp.ndarray           Input array of discretized parameters for simulation model 
            Y : jnp.ndarray           Output values of the simulation model
            Xs : jnp.ndarray          The left-out row of X  
            Ys : jnp.ndarray          The left-out row of Y
            x : jnp.ndarray           Vector of kernel hyperparameters to use when evaluating the kernel
            row_num : int             Row number to leave out for leave-one-out cross validation.
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
        K_inv = jnp.linalg.inv(self.kernel(X, X, x, cache_key = f"XX{row_num_str}"))
        kernel_Xs_Xs = self.kernel(Xs, Xs, x, cache_key = f"XsXs{row_num_str}")
        kernel_X_Xs = self.kernel(X, Xs, x, cache_key = f"XXs{row_num_str}")
        
        # For most kernels kernel_Xs_X and kernel_X_Xs are the same
        if self.kernel_spec != "GRE":            
            kernel_Xs_X = kernel_X_Xs.T 
        else:
            kernel_Xs_X = self.kernel(Xs, X, x, cache_key = f"XsX{row_num_str}")

        # Basis functions
        # A = m x d (sparse matrix)
        # X = n x d
        # Xs = n_test x d
        VA = x2fx(X, self.sparse_basis)
        vAT = x2fx(Xs, self.sparse_basis).T

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
        #inv_VA_T_at_K_inv_at_VA = jnp.linalg.inv(VA_T_at_K_inv @ VA)
        inv_VA_T_at_K_inv_at_VA = jnp.linalg.pinv(VA_T_at_K_inv @ VA, hermitian = True)
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

    
    def cv_loss(self, x : jnp.ndarray) -> float:
        """
        Calculate the loss (log-likelihood of test data) using leave-one-out cross validation (LOOCV).
      
        Parameters:  
            x : jnp.ndarray           Vector of kernel hyperparameters to use when evaluating the kernel
        Returns:
            float              
        """

        # A = m x d (sparse matrix)
        # X = n_train x d
        # Y = n_train x 1
        # x = p x 1
        return sum(
            self.cv_local_loss(                   
                x,
                i
            ) for i in range(self.X_normalised.shape[0])
        )

    def set_normalised_data(self, X, Y):
        """
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

    def set_sparse_basis(self, A):
        """
        Parameters:
            A : jnp.ndarray             shape (m, d), binary matrix representing the sparse basis     
        
        Returns:
            None    
        """
        self.sparse_basis = A
    
    def perform_extrapolation(self, x : jnp.ndarray, return_mu_and_var : bool = False) -> dict:
        """
        Perform Sparse Probabilistic Richardson Extrapolation (SPRE) for the given values of
        the kernel parameters.
        
        Parameters:
            x : jnp.ndarray             shape (p,), kernel parameters
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
    
        # Define the gradient function
        gradient_function = grad(self.cv_loss)

        # Evaluate gradient at x
        gradient = gradient_function(x)

        # Evaluate cv
        cv = self.cv_loss(x)

        # Output
        out = {
            "cv": cv,
            "cv_grad": jnp.array(gradient)
        }

        #debug.print("cv = {}, grad = {}, x = {}", cv, gradient, x)

        # Add extra ouput if requested
        if return_mu_and_var:
            # LOOCV predictions
            mu_cv = jnp.zeros(self.X_normalised.shape[0])
            var_cv = jnp.zeros(self.X_normalised.shape[0])
            for i in range(self.X_normalised.shape[0]):               
                mu_val, cov_val = self.cv_local_loss(x, i, return_mu_cov = True)               
                mu_cv = mu_cv.at[i].set((self.nY * mu_val[0]))              
                var_cv = var_cv.at[i].set((self.nY**2 * cov_val[0][0]))
              
            # Output
            mu_value, cov_value = self.cv_loss_calculation(self.X_normalised, self.Y_normalised, jnp.zeros((1, self.dimension)), jnp.zeros((1, self.dimension)), x, return_mu_cov = True)
            out0 = out
            out = {
                "mu": self.nY * mu_value,
                "var": self.nY**2 * cov_value,           
                "mu_cv": mu_cv,
                "var_cv": var_cv            
            }
            out.update(out0)
            
        return out

    def objective(self, x : jnp.ndarray) -> tuple:
        """
        Objective function used to fit the hyperparameters of the kernel.
        
        Parameters:
            x : jnp.ndarray             shape (p,), kernel hyperparameters
        Returns:
            tuple
        """

        # Return the negative log likelihood using LOOCV with gradient
        out = self.perform_extrapolation(x)        
        return -out['cv'], -out['cv_grad']
        
    def perform_extrapolation_optimization(self) -> dict:
        """
        Optimize kernel hyperparameters for SPRE.

        Parameters:
           None
          
        Returns:
            out : dict with keys:
                 x  = p x 1, fitted kernel hyperparameters 
                 cv = scalar, LOOCV criterion
        """
   
        # Set up the cach with values to use
        self.set_kernel_cache()

        # Set up the solver to use
        solver = GradientDescent(fun = self.objective, maxiter=100, value_and_grad = True, stepsize=1e-3)#, tol=1e-3)
        
        # Fit the best hyperparameters for the kernel
        result = solver.run(jnp.array(self.default_kernel_parameters))

        # Evaluate the final LOOCV negative log likelihood result 
        result_value, _ = self.objective(result.params)

        return {
            'x'  : result.params,
            'cv' : result_value
        }

    def stepwise_selection(self) -> dict:
        """
        Stepwise model selection for SPRE.

        Parameters:
            X       : jnp.ndarray           Training inputs, (n_train, d)
            Y       : jnp.ndarray           Training outputs, (n_train,)
            k_name  : str                   Kernel name ("Gaussian", "GaussianARD", "Matern1/2", "Matern3/2", "white")

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

        # Initialise with just an intercept
        A = jnp.zeros((1, self.dimension), dtype=int)  
        self.set_sparse_basis(A)

        # Handle selection differently if doing GRE
        if self.gre_base is not None:
            return self._GRE_stepwise_selection()
        
        order = 0
        fit = self.perform_extrapolation_optimization()
    
        cv = fit['cv']

        carry_on = 0

        while carry_on:
            order += 1
            A_extra = stepwise(A, order)  # Generate new predictors of given order
            n_extra = A_extra.shape[0]
            to_include = jnp.zeros(n_extra, dtype=bool)

            print(f"Fitting interactions of order {order}:")

            for i in tqdm(range(n_extra), desc="Stepwise progress"):
                A_new = jnp.vstack([A, A_extra[i]])
                self.set_sparse_basis(A_new)
                fit_new = self.perform_extrapolation_optimization()
                cv_new = fit_new['cv']
                if cv_new < cv:
                    to_include[i] = True

            if jnp.any(to_include):
                A_updated = jnp.vstack([A, A_extra[to_include]])
                self.set_sparse_basis(A_updated)
                fit_updated = self.perform_extrapolation_optimization()
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
        out = self.perform_extrapolation(x_opt, return_mu_and_var = True)

        return out
    
    def _GRE_stepwise_selection(self) -> dict:
        """
        Stepwise model selection for GRE (Gauss-Richardson Extrapolation).

        Parameters:
            None

        Returns:
            out : dict
                Dictionary with predictive mean, variance, and fitted model details.
        """
       
        # initial rate function: only intercept
        B = jnp.zeros((1, self.dimension), dtype=int)    
        order = 0

        fit = self.perform_extrapolation_optimization()
        cv = fit["cv"]

        carry_on = True
        while carry_on:
            order += 1
            B_extra = stepwise(B, order)  # Generate all predictors of the next order
            n_extra = B_extra.shape[0]

            print(f"Fitting interactions of order {order}...")

            to_include = jnp.zeros(n_extra, dtype=bool)
            for i in range(n_extra):
                B_new = jnp.vstack([B, B_extra[i, :]])
                self.set_kernel_spec(self.kernel_base, B_new)
                fit_new = self.perform_extrapolation_optimization()
                cv_new = fit_new["cv"]

                if cv_new < cv:                    
                    to_include = to_include.at[i].set(True)

                # Progress bar substitute
                #cwbar((i + 1) / n_extra)

            if jnp.any(to_include):
                B_updated = jnp.vstack([B, B_extra[to_include, :]])
                self.set_kernel_spec(self.kernel_base, B_updated)
                fit_updated = self.perform_extrapolation_optimization()
                cv_updated = fit_updated["cv"]

                if cv_updated >= cv:
                    carry_on = False
                else:
                    fit = fit_updated
                    B = B_updated
                    cv = cv_updated
            else:
                carry_on = False

            #cwbar("done")

        # Final GP fit with optimal parameters
        x_opt = fit["x"]
        out = self.perform_extrapolation(x_opt, return_mu_and_var=True)

        return out
    