# Python modules
import jax.numpy as jnp
from jax import grad, debug
from jaxopt import GradientDescent
from tqdm import tqdm  # For progress bars
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Application modules
from helper_functions import x2fx, softplus, cellsum, white, remove_row, stepwise

class SPRE:
    '''
    Sparse Probabilistic Richardson Extrapolation (SPRE).
    '''

    def __init__(self, kernel_spec : str, dimension : int, gre_base : jnp.ndarray = None):
        '''
        Sets up the SPRE class with an initial kernel for extrapolation.

        Parameters:  
            kernel_spec : str               Name of the kernel to set up and use.
            dimension : int                 Data dimension.
            gre_base : jnpndarray            Basis for compatability layer for GRE.          
        Returns:
            None         
        '''
        
        # Smallest allowed amplitude, epsilson, used in the kernel.
        self.ep = 1e-16  

        # Set dimension
        self.dimension = dimension

        # Set up the kernel to use
        self.set_kernel_spec(kernel_spec, gre_base)
       
        # Set kernel cache
        self.kernel_cache = {}

    def cdist_jax(self, XA, XB):
        """
        Computes pairwise Euclidean distances between two sets of vectors (rows of XA and XB).
        Equivalent to scipy.spatial.distance.cdist(XA, XB, 'euclidean')
        """
        # ||a - b||² = ||a||² + ||b||² - 2a·b
        XA_sq = jnp.sum(XA ** 2, axis=1, keepdims=True)  # (m, 1)
        XB_sq = jnp.sum(XB ** 2, axis=1)  # (n,)
        cross_term = jnp.dot(XA, XB.T)  # (m, n)
        
        dists = jnp.sqrt(XA_sq - 2 * cross_term + XB_sq)  
        return dists

    def set_kernel_spec(self, kernel_spec : str, gre_base : jnp.ndarray = None):
        '''
        Sets up the kernel function to use for analysis, such that kernal_function(X1, X2, x)
        where X1 and X2 are simulation data output and x is array of hyperparameters for the kernel.
            X1 = n1 x dimension
            X2 = n2 x dimension
            x = p x 1
        Where n1 and n2 are the number of observations and p the number of hyper parameters.

        Parameters:  
            kernel_spec : str               Name of the kernel to set up and use         
            gre_base : jnp.ndarray          Basis for compatability layer for GRE             
        Returns:
            None         
        '''

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
        self.set_kernel_default_parameters(self.kernel_spec) 
       
    def set_kernel_default_parameters(self, kernel_spec : str):
         
        # Set default parameters
        match kernel_spec:
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
                self.set_kernel_default_parameters(self.kernel_base)
                base_default_parameters = self.default_kernel_parameters
                self.default_kernel_parameters = [1.0] 
                self.default_kernel_parameters.extend(base_default_parameters)                  
            case _:
                raise ValueError(f"Unknown kernel specification: {kernel_spec}")
            
    def set_kernel_cache(self):

        self.kernel_cache = {}

        for i in range(self.X_normalised.shape[0]):
            X = remove_row(self.X_normalised, i)           
            Xs = self.X_normalised[i:(i+1), :]
            self.kernel_cache[f"XX{i}"] = self.calculate_kernel_cached_bit(X, X)         
            self.kernel_cache[f"XXs{i}"] = self.calculate_kernel_cached_bit(X, Xs) 
            self.kernel_cache[f"XsXs{i}"] = self.calculate_kernel_cached_bit(Xs, Xs)        
            self.kernel_cache[f"XsX{i}"] = self.calculate_kernel_cached_bit(Xs, X) 
              
    def calculate_kernel_cached_bit(self, X1, X2):
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

    def get_kernel_cached_bit(self, X1, X2, cache_key):
        if cache_key is not None and cache_key in self.kernel_cache:
            return self.kernel_cache[cache_key]   
        else:
            return self.calculate_kernel_cached_bit(X1, X2)
       
    def kernel(self, X1, X2, x = None, cache_key = None):

        if x is None:
            x = self.default_kernel_parameters

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

    # Basis functions
    # A = m x d
    # X = n x d
    # Xs = n_test x d
    def V(self, X):
        return x2fx(X, self.sparse_basis)  # n x m

    def v(self, Xs):
        return x2fx(Xs, self.sparse_basis).T  # m x n_test

    # Residual term
    # A = m x d
    # X = n_train x d
    # Xs = n_test x d
    # x = p x 1
    def residual(self, X, Xs, x):
        K_inv = jnp.linalg.inv(self.kernel(X, X, x))
        return self.v(Xs) - self.V(X).T @ K_inv @ self.kernel(X, Xs, x)
        
    # Coefficient estimator
    # A = m x d
    # X = n_train x d
    # Y = n_train x 1
    # x = p x 1
    def beta(self, X, Y, x):
        K_inv = jnp.linalg.inv(self.kernel(X, X, x))
        VA = self.V(X)
        return jnp.linalg.pinv(VA.T @ K_inv @ VA, hermitian = True) @ (VA.T @ K_inv @ Y)


    def mu_GP(self, X, Y, Xs, x):
        #function R^d -> R, predictive mean for fitted GP
        K_inv = jnp.linalg.inv(self.kernel(X, X, x))

        ############
        pt1 = self.kernel(Xs, X, x) @ K_inv @ Y
        pt2 = self.residual(X, Xs, x).T @ self.beta(X, Y, x)
        pt21 = self.residual(X, Xs, x).T
        pt22 = self.beta(X, Y, x)
        print(f"pt1 = {pt1}")
        print(f"pt2 = {pt2}")
        print(f"pt21 = {pt21}")
        print(f"pt22 = {pt22}\n")
        K_inv = jnp.linalg.inv(self.kernel(X, X, x))
        VA = self.V(X)
        #inv( V(A,X)' * inv(k(X,X,x)) * V(A,X) );
        
        pt30 = VA.T @ K_inv @ VA
        #pt30 = jnp.round(pt30,4)
        print(f"pt30 = {pt30}")
        pt3 = jnp.linalg.inv(pt30)
        pt3_ps = jnp.linalg.pinv(pt30, hermitian = True)#, rcond = 1e-16)
        lin_check = jnp.linalg.cond(pt30)
        print(f"Cond = {lin_check}")
        #pt3 = jnp.linalg.inv(pt30)        
        #pt3 = jnp.linalg.solve(pt30, jnp.identity(pt30.shape[0]))
        pt4 = (VA.T @ K_inv @ Y)
        print(f"pt3 = {pt3}")
        print(f"pt3_ps = {pt3_ps}")
        pt3Check = pt30 @ pt3
        print(f"pt3Check = {pt3Check}")
        pt3Check_ps = pt30 @ pt3_ps
        print(f"pt3Check_ps = {pt3Check_ps}")
        print(f"pt4 = {pt4}\n")
        pt31 = VA
        pt32 = K_inv
        print(f"pt31 = {pt31}")
        print(f"pt32 = {pt32}\n")
        #################

        return self.kernel(Xs, X, x) @ K_inv @ Y + self.residual(X, Xs, x).T @ self.beta(X, Y, x)
  
    def cov_GP(self, X, Xs, x):
        # function R^d x R^d -> R, predictive covariance for fitted GP
        K_inv = jnp.linalg.inv(self.kernel(X, X, x))
        VA = self.V(X)
        residual_X_Xs = self.residual(X, Xs, x)
        return (self.kernel(Xs, Xs, x)
                - self.kernel(Xs, X, x) @ K_inv @ self.kernel(X, Xs, x)
                + residual_X_Xs.T @ jnp.linalg.inv(VA.T @ K_inv @ VA) @ residual_X_Xs)

    # Cross-validation local loss (log-likelihood of test data)
    def cv_local_loss(self, x, row_num, return_info = False):
        # Cross-validation local loss (log-likelihood of test data)
        # A = m x d
        # X = n_train x d
        # Y = n_train x 1
        # Xs = n_test x d
        # Ys = n_test x 1
        # x = p x 1

        X = remove_row(self.X_normalised, row_num)
        Y = remove_row(self.Y_normalised, row_num)
        Xs = self.X_normalised[row_num:(row_num+1), :]
        Ys = self.Y_normalised[row_num:(row_num+1)]

        # Calculate some bits firstly
        
        K_inv = jnp.linalg.inv(self.kernel(X, X, x, cache_key = f"XX{row_num}"))
        kernel_X_Xs = self.kernel(X, Xs, x, cache_key = f"XXs{row_num}")
        
        if self.kernel_spec != "GRE":            
            kernel_Xs_X = kernel_X_Xs.T 
        else:
            kernel_Xs_X = self.kernel(Xs, X, x, cache_key = f"XsX{row_num}")

      
        # Basis functions
        # A = m x d
        # X = n x d
        # Xs = n_test x d
        VA = x2fx(X, self.sparse_basis)
        vAT = x2fx(Xs, self.sparse_basis).T

        # Residual term
        # A = m x d
        # X = n_train x d
        # Xs = n_test x d
        # x = p x 1   
        VA_T_at_K_inv = VA.T @ K_inv  
        residual_X_Xs = vAT - VA_T_at_K_inv @ kernel_X_Xs
    
        # Predictive covariance
        # A = m x d
        # X = n_train x d
        # Xs = n_test x d
        # x = p x 1  
        #inv_VA_T_at_K_inv_at_VA = jnp.linalg.inv(VA_T_at_K_inv @ VA)
        inv_VA_T_at_K_inv_at_VA = jnp.linalg.pinv(VA_T_at_K_inv @ VA, hermitian = True)
        cov_val = (self.kernel(Xs, Xs, x, cache_key = f"XsXs{row_num}")
                - kernel_Xs_X @ K_inv @ kernel_X_Xs
                + residual_X_Xs.T @ inv_VA_T_at_K_inv_at_VA @ residual_X_Xs)
        
        # Coefficient estimator, beta
        # A = m x d
        # X = n_train x d
        # Y = n_train x 1
        # x = p x 1          
        beta_X_Y = inv_VA_T_at_K_inv_at_VA @ (VA_T_at_K_inv @ Y)
        
        # Predictive mean
        # A = m x d
        # X = n_train x d
        # Y = n_train x 1
        # Xs = n_test x d
        # x = p x 1
        mu_val = kernel_Xs_X @ K_inv @ Y + residual_X_Xs.T @ beta_X_Y

        diff = Ys - mu_val
        inv_cov = jnp.linalg.inv(cov_val)
        term1 = -0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * cov_val))
        term2 = -0.5 * diff.T @ inv_cov @ diff
        
        return term1 + term2

    # LOOCV loss 
        # A = m x d
        # X = n_train x d
        # Y = n_train x 1
        # x = p x 1
    def cv_loss(self, x):
        return sum(
            self.cv_local_loss(                   
                x,
                i
            ) for i in range(self.X_normalised.shape[0])
        )

    def set_normalised_data(self, X, Y):
        '''
        Parameters:
            A : jnp.ndarray             shape (m, d), binary matrix representing the sparse basis
            X : jnp.ndarray             shape (n_train, d), training inputs
            Y : jnp.ndarray             shape (n_train,), training outputs

        Returns:
            None
        '''
          
        # Data normalization   
        self.nX = self.ep + (jnp.max(X, axis=0) - jnp.min(X, axis=0))
        self.nY = self.ep + (jnp.max(Y) - jnp.min(Y))
        self.X_normalised = X / self.nX
        self.Y_normalised = Y / self.nY

    def set_sparse_basis(self, A):
        '''
        Parameters:
            A : jnp.ndarray             shape (m, d), binary matrix representing the sparse basis     
        
        Returns:
            None    
        '''
        self.sparse_basis = A

    # Define the function to calculate gradient from
    #def gradient_eval(self, x):
    #        return self.cv_loss(x)
    
    def perform_extrapolation(self, x : jnp.ndarray, return_mu_and_var : bool = False):
        """
        Sparse Probabilistic Richardson Extrapolation (SPRE).
        
        Parameters:
            x : jnp.ndarray             shape (p,), kernel parameters
            return_mu_and_var : bool  Whether to return variables: mu, mu_cv, var and var_cv

        Returns:
            out : dict with keys:
                - mu: scalar, predictive mean for f(0)
                - var: scalar, predictive variance for f(0)
                - mu_cv: n_train x 1, LOOCV predictive means
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

        debug.print("cv = {}, grad = {}, x = {}", cv, gradient, x)

        # Add extra ouput if requested
        if return_mu_and_var:
            # LOOCV predictions
            mu_cv = jnp.array([
                self.nY * self.mu_GP(remove_row(self.X_normalised, i), remove_row(self.Y_normalised, i), self.X_normalised[i:i+1, :], x)
                for i in range(self.X_normalised.shape[0])
            ])

            var_cv = jnp.array([
                self.nY**2 * self.cov_GP(remove_row(self.X_normalised, i), self.X_normalised[i:i+1, :], x)
                for i in range(self.X_normalised.shape[0])
            ]).flatten()

            # Output
            out0 = out
            out = {
                "mu": self.nY * self.mu_GP(self.X_normalised, self.Y_normalised, jnp.zeros((1, self.dimension)), x),
                "var": self.nY**2 * self.cov_GP(self.X_normalised, jnp.zeros((1, self.dimension)), x),           
                "mu_cv": mu_cv,
                "var_cv": var_cv            
            }
            out.update(out0)
            

        return out

    def objective(self, x):
        # Objective function
        # LOOCV (negative log likelihood of held-out datum)
        out = self.perform_extrapolation(x)        
        return -out['cv'], -out['cv_grad']
        
    def perform_extrapolation_optimization(self):
        """
        Optimize kernel parameters for SPRE.

        Parameters:
           None
          
        Returns:
            out : dict with keys:
                - x: p x 1, fitted kernel parameters 
                - cv: scalar, LOOCV criterion
        """
   
        '''
        # Optimization
        result = minimize(
            fun = objective,
            x0 = x0,  
            #method = 'BFGS',
            #tol = 1e-3
            #options={
            #    'maxiter': 10 # set to 10 for speed
            #}
        )
    
        print(result)
        return {
            'x': result.x,
            'cv': result.fun
        }
        '''
        
        self.set_kernel_cache()

        solver = GradientDescent(fun = self.objective, maxiter=100, value_and_grad = True, stepsize=1e-3)#, tol=1e-3)
        
        result = solver.run(jnp.array(self.default_kernel_parameters))
        result_value, _ = self.objective(result.params)

        return {
            'x'  : result.params,
            'cv' : result_value
        }
        

        '''
        # Optimization
        result = minimize(
            fun = objective,
            x0 = x0,  
            method = 'BFGS',
            tol = 1e-3
            #options={
            #    'maxiter': 10 # set to 10 for speed
            #}
        )
    
        print(result)
        return {
            'x': result.x,
            'cv': result.fun
        }
        '''

    def stepwise_selection(self):
        """
        Stepwise model selection for SPRE.

        Parameters:
            X       : ndarray of shape (n_train, d), training inputs
            Y       : ndarray of shape (n_train,), training outputs
            k_name  : str, kernel name ("Gaussian", "GaussianARD", "Matern1/2", "Matern3/2", "white")

        Returns:
            out     : dict, result of SPRE using optimal model
                    % out.mu      = scalar, predictive mean for f(0)
                    % out.cov     = scalar, predictive variance for f(0)
                    % out.mu_GP   = function R^d -> R, predictive mean for fitted GP
                    % out.cov_GP  = function R^d x R^d -> R, predictive covariance for fitted GP
                    % out.mu_cv   = n_train x 1, LOOCV predictive means
                    % out.var_cv  = n_train x 1, LOOCV predictive variances
                    % out.cv      = scalar, LOOCV criterion
                    % out.cv_grad = p x 1, gradient of LOOCV criterion
        """

        A = jnp.zeros((1, self.dimension), dtype=int)  # Initialise with just an intercept
        #A = jnp.array([[0, 0]])
        self.set_sparse_basis(A)

        # Handle selection differently if doing GRE
        if self.gre_base is not None:
            return self._GRE_stepwise_selection()
        
        #A = jnp.array([[0, 0], [0, 1], [1, 1], [2, 0]])
        order = 0
        fit = self.perform_extrapolation_optimization()
        #print("stepwise")
        #print(A)
        #print(X)
        #print(Y)
        #print(k_name)
        #print(fit)
        cv = fit['cv']

        carry_on = 0 # True

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
        print(x_opt)
        #print(A)
        #print(X)
        #print(Y)
        # Final model with best kernel parameters and basis A
        out = self.perform_extrapolation(x_opt, return_mu_and_var=True)

        return out
    
    def _GRE_stepwise_selection(self):
        """
        Stepwise model selection for GRE (Gauss-Richardson Extrapolation).

        Parameters:
            None

        Returns:
            out : dict
                Dictionary with predictive mean, variance, and fitted model details.
        """
       
        B = jnp.zeros((1, self.dimension), dtype=int)    # initial rate function: only intercept
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
                    to_include[i] = True

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
    