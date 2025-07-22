# Python modules
import jax.numpy as jnp
from jax import grad, debug
from jaxopt import GradientDescent

# Application modules
from helper_functions import x2fx, softplus, cellsum, white, remove_row

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
        self.set_kernel(kernel_spec, gre_base)

    def cdist_jax(self, XA, XB):
        """
        Computes pairwise Euclidean distances between two sets of vectors (rows of XA and XB).
        Equivalent to scipy.spatial.distance.cdist(XA, XB, 'euclidean')
        """
        # ||a - b||² = ||a||² + ||b||² - 2a·b
        XA_sq = jnp.sum(XA ** 2, axis=1, keepdims=True)  # (m, 1)
        XB_sq = jnp.sum(XB ** 2, axis=1)  # (n,)
        cross_term = jnp.dot(XA, XB.T)  # (m, n)
        
        dists = jnp.sqrt(XA_sq - 2 * cross_term + XB_sq)  # broadcasting
        return dists

    def _create_kernel(self, kernel_spec : str):
        '''
        Creates the kernel function to use for analysis, such that kernal_function(X1, X2, x)
        where X1 and X2 are simulation data output and x is array of hyperparameters for the kernel.
            X1 = n1 x dimension
            X2 = n2 x dimension
            x = p x 1
        Where n1 and n2 are the number of observations and p the number of hyper parameters.

        Parameters:  
            kernel_spec : str               Name of the kernel to set up and use

        Returns:
            function         
        '''

        if kernel_spec == "Gaussian":
            # Set default parameters for Gaussian kernel
            self.default_kernel_parameters = [1.0, 0.1]
            def kernal_function(X1, X2, x = self.default_kernel_parameters):
                return (self.ep + softplus(x[0])) * jnp.exp(-self.cdist_jax(X1, X2) ** 2 / softplus(x[1])**2)

        elif kernel_spec == "GaussianARD":            
            # Set default parameters GaussianARD kernel
            self.default_kernel_parameters = [1.0] 
            self.default_kernel_parameters.extend([0.1] * self.dimension)
            def kernal_function(X1, X2, x = self.default_kernel_parameters):
                amp = self.ep + softplus(x[0])
                lengthscales = [self.cdist_jax(X1[:, [i]], X2[:, [i]])**2 / softplus(x[i+1])**2 for i in range(self.dimension)]
                return amp * jnp.exp(-cellsum(lengthscales))
            
        elif kernel_spec == "white":
            # Set default parameters for white kernel
            self.default_kernel_parameters = [1.0] 
            def kernal_function(X1, X2, x = self.default_kernel_parameters):
                return (self.ep + softplus(x[0])) * white(X1, X2)

        elif kernel_spec == "Matern1/2":
            # Set default parameters for Matern1/2 kernel
            self.default_kernel_parameters = [1.0, 1.0] 
            def kernal_function(X1, X2, x = self.default_kernel_parameters):
                return (self.ep + softplus(x[0])) * jnp.exp(-self.cdist_jax(X1, X2) / softplus(x[1]))

        elif kernel_spec == "Matern3/2":
            # Set default parameters for Matern3/2 kernel
            self.default_kernel_parameters = [1.0, 1.0]
            def kernal_function(X1, X2, x = self.default_kernel_parameters):
                r = self.cdist_jax(X1, X2)
                l = softplus(x[1])
                sqrt3_r_l = jnp.sqrt(3) * r / l
                return (self.ep + softplus(x[0])) * (1 + sqrt3_r_l) * jnp.exp(-sqrt3_r_l)
                 
        else:
            raise ValueError(f"Unknown kernel specification: {kernel_spec}")

        # Return kernel function 
        return kernal_function

    def set_kernel(self, kernel_spec : str, gre_base : jnp.ndarray = None):
        '''
        Sets up the kernel function to use for analysis, such that kernal_function(X1, X2, x)
        where X1 and X2 are simulation data output and x is array of hyperparameters for the kernel.
            X1 = n1 x dimension
            X2 = n2 x dimension
            x = p x 1
        Where n1 and n2 are the number of observations and p the number of hyper parameters.

        Parameters:  
            kernel_spec : str               Name of the kernel to set up and use         
            gre_base : jnp.ndarray            Basis for compatability layer for GRE             
        Returns:
            None         
        '''
         
        if gre_base is None:
            # Create kernal function
            kernel_function = self._create_kernel(kernel_spec)
        
        else: 
            # Compatability layer for GRE 
             
            # Create base kernel 
            kernel_base = self._create_kernel(kernel_spec)

            # Set default parameters for x for base, and then new kernal function         
            default_base_paras = [1.0]
            self.default_kernel_parameters = default_base_paras.extend(self.default_kernel_parameters)      

            # Define kernel function
            def kernel_function(X1, X2, x = self.default_kernel_parameters):
                amp = self.ep + softplus(x[0])
                # Convergence rate ansatz b(x)
                base_X1 = jnp.sum(x2fx(X1, gre_base), axis=1)
                base_X2 = jnp.sum(x2fx(X2, gre_base), axis=1)
                return amp * base_X1[:, None] * kernel_base(X1, X2, x = x[1:]) * base_X2[None, :]
               
        # Store kernel function in class for later use
        self.kernel = kernel_function

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
        return jnp.linalg.inv(VA.T @ K_inv @ VA) @ (VA.T @ K_inv @ Y)

    # Predictive mean
        # A = m x d
        # X = n_train x d
        # Y = n_train x 1
        # Xs = n_test x d
        # x = p x 1
    def mu_GP(self, X, Y, Xs, x):
        #function R^d -> R, predictive mean for fitted GP
        K_inv = jnp.linalg.inv(self.kernel(X, X, x))
        return self.kernel(Xs, X, x) @ K_inv @ Y + self.residual(X, Xs, x).T @ self.beta(X, Y, x)

    # Predictive covariance
        # A = m x d
        # X = n_train x d
        # Xs = n_test x d
        # x = p x 1
    def cov_GP(self, X, Xs, x):
        # function R^d x R^d -> R, predictive covariance for fitted GP
        K_inv = jnp.linalg.inv(self.kernel(X, X, x))
        VA = self.V(X)
        residual_X_Xs = self.residual(X, Xs, x)
        return (self.kernel(Xs, Xs, x)
                - self.kernel(Xs, X, x) @ K_inv @ self.kernel(X, Xs, x)
                + residual_X_Xs.T @ jnp.linalg.inv(VA.T @ K_inv @ VA) @ residual_X_Xs)

    # Cross-validation local loss (log-likelihood of test data)
        # A = m x d
        # X = n_train x d
        # Y = n_train x 1
        # Xs = n_test x d
        # Ys = n_test x 1
        # x = p x 1
    def cv_local_loss(self, X, Y, Xs, Ys, x):
        #cov_val = self.cov_GP(X, Xs, x)
        #mu_val = self.mu_GP(X, Y, Xs, x)
        # Calculate cov_GP
        K_inv = jnp.linalg.inv(self.kernel(X, X, x))
        VA = self.V(X)
        #Calculate residual     
        kernel_X_Xs = self.kernel(X, Xs, x)
        residual_X_Xs = self.v(Xs) - self.V(X).T @ K_inv @ kernel_X_Xs
    
        # Calculate cov_GP
        kernel_Xs_X = self.kernel(Xs, X, x)
        cov_val = (self.kernel(Xs, Xs, x)
                - kernel_Xs_X @ K_inv @ kernel_X_Xs
                + residual_X_Xs.T @ jnp.linalg.inv(VA.T @ K_inv @ VA) @ residual_X_Xs)
        
        # Calculate beta 
        beta_X_Y = jnp.linalg.inv(VA.T @ K_inv @ VA) @ (VA.T @ K_inv @ Y)
    
        # Calculate mu_GP
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
                remove_row(self.X_normalised, i),
                remove_row(self.Y_normalised, i),
                self.X_normalised[i:i+1, :],
                self.Y_normalised[i:i+1],
                x
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
            out.update({
                "mu": self.nY * self.mu_GP(self.X_normalised, self.Y_normalised, jnp.zeros((1, self.dimension)), x),
                "var": self.nY**2 * self.cov_GP(self.X_normalised, jnp.zeros((1, self.dimension)), x),           
                "mu_cv": mu_cv,
                "var_cv": var_cv            
            })

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
