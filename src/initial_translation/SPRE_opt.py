# Python modules
import jax.numpy as jnp
#from jax.scipy.optimize import minimize
from jaxopt import GradientDescent

# Application modules
from src.initial_translation.kernel import kernel, get_default_args
from src.initial_translation import SPRE


def SPRE_opt(A, X, Y, str_):
    """
    Optimize kernel parameters for SPRE.

    Parameters:
        A : jnp.ndarray,
            m x d, binary matrix representing the sparse basis 
        X : jnp.ndarray,
            n_train x d, training inputs
        Y : jnp.ndarray,
            n_train x 1, training outputs
        str_ : kernel specification (or (B, kernel_name))

    Returns:
        out : dict with keys:
            - x: p x 1, fitted kernel parameters 
            - cv: scalar, LOOCV criterion
    """

    d = X.shape[1]

    # Get initial default parameters from kernel for x
    x0 = jnp.array(get_default_args(kernel(str_, d))['x'])

    # Objective function
    # LOOCV (negative log likelihood of held-out datum)
    def objective(x):
        out = SPRE.SPRE(A, X, Y, x, str_)
        return -out['cv'] #, -out['cv_grad']
    
    solver = GradientDescent(fun = objective, maxiter=200, implicit_diff = False, tol = 0.01)
    result = solver.run(x0)
    result_value = objective(result.params)

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
   

    return {
        'x': result.x,
        'cv': result.fun
    }
    '''
