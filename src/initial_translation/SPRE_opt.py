import numpy as np
from scipy.optimize import minimize

def SPRE_opt(A, X, Y, str_):
    """
    Optimize kernel parameters for SPRE.

    Parameters:
        A    : np.ndarray, shape (m, d), sparse basis
        X    : np.ndarray, shape (n_train, d), training inputs
        Y    : np.ndarray, shape (n_train,), training outputs
        str_ : kernel specification (or (B, kernel_name))

    Returns:
        out : dict with keys:
            - x: optimized kernel parameters
            - cv: LOOCV criterion (negative log-likelihood)
    """

    d = X.shape[1]

    # Initial parameters from kernel
    _, x0 = kernel(str_, d)

    # Objective function: returns value and gradient
    def objective(x):
        out = SPRE(A, X, Y, x, str_)
        return -out['cv'], -out['cv_grad']

    # Optimization
    result = minimize(
        fun=objective,
        x0=x0,
        method='trust-constr',
        jac=True,
        options={
            'disp': False,
            'maxiter': 10
        }
    )

    return {
        'x': result.x,
        'cv': -result.fun
    }
