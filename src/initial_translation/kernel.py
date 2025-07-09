

# Python modules
import numpy as np
from scipy.spatial.distance import cdist
from inspect import signature

# Application modules
from src.initial_translation.helper_functions import x2fx, softplus, cellsum, white

def kernel(spec, d):
    """
    Kernel constructor.

    Parameters:
        spec : str or tuple (B, base_kernel)
            Kernel specification string or GRE composite.
        d    : int
            Data dimension.

    Returns:
        kernal_function : function
            kernel function kernal_function(X1, X2, x)
        x0 : np.ndarray
            default parameter values for the kernel
    """
    ep = 1e-16  # Smallest allowed amplitude

    if spec == "Gaussian":
        # X1 = n1 x d
        # X2 = n2 x d
        # x = p x 1
        def kernal_function(X1, X2, x = [1.0, 0.1]):
            return (ep + softplus(x[0])) * np.exp(-cdist(X1, X2) ** 2 / softplus(x[1])**2)

    elif spec == "GaussianARD":
        # X1 = n1 x d
        # X2 = n2 x d
        # x = p x 1
        x0 = [1.0]
        x0.extend([0.1] * d)
        def kernal_function(X1, X2, x = x0):
            amp = ep + softplus(x[0])
            lengthscales = [cdist(X1[:, [i]], X2[:, [i]])**2 / softplus(x[i+1])**2 for i in range(d)]
            return amp * np.exp(-cellsum(lengthscales))
        
    elif spec == "white":
        # X1 = n1 x d
        # X2 = n2 x d
        # x = p x 1
        def kernal_function(X1, X2, x = 1.0):
            return (ep + softplus(x)) * white(X1, X2)

    elif spec == "Matern1/2":
        # X1 = n1 x d
        # X2 = n2 x d
        # x = p x 1
        def kernal_function(X1, X2, x = [1.0, 1.0]):
            return (ep + softplus(x[0])) * np.exp(-cdist(X1, X2) / softplus(x[1]))

    elif spec == "Matern3/2":
        # X1 = n1 x d
        # X2 = n2 x d
        # x = p x 1
        def kernal_function(X1, X2, x = [1.0, 1.0]):
            r = cdist(X1, X2)
            l = softplus(x[1])
            sqrt3_r_l = np.sqrt(3) * r / l
            return (ep + softplus(x[0])) * (1 + sqrt3_r_l) * np.exp(-sqrt3_r_l)

    elif isinstance(spec, tuple):  # compatability layer for GRE
        # basis functions
        # B = m x d
        B, k_name = spec

        # convergence rate ansatz b(x)
        def base(X):
            return np.sum(x2fx(X, B), axis=1)
        
        k_base = kernel(k_name, d)

        # Get default values for x for base, and then new kernal function
        x0_base = signature(k_base).parameters['x'].default
        x0 = [1.0]
        x0.extend(x0_base)      

        # X1 = n1 x d
        # X2 = n2 x d
        # x = p x 1
        def kernal_function(X1, X2, x = x0):
            amp = ep + softplus(x[0])
            return amp * base(X1)[:, None] * k_base(X1, X2, x = x0[1:]) * base(X2)[None, :]
        
    else:
        raise ValueError(f"Unknown kernel specification: {spec}")

    return kernal_function