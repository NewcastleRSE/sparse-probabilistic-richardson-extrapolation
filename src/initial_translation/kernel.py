import numpy as np
from scipy.spatial.distance import cdist

#def softplus(x):
#    return np.log1p(np.exp(x))  # numerically stable

def white(X1, X2):
    # Returns 1 if rows of X1 and X2 are equal (white noise kernel)
    return (cdist(X1, X2) == 0).astype(float)

#def cellsum(arrays):
#    return sum(arrays)

def x2fx(X, B):
    # Applies basis function transformation given basis exponents B (m x d)
    return np.prod([X[:, i:i+1] ** B[:, i] for i in range(B.shape[1])], axis=0)

def kernel(spec, d):
    """
    Kernel constructor.

    Parameters:
        spec : str or tuple (B, base_kernel)
            Kernel specification string or GRE composite.
        d    : int
            Data dimension.

    Returns:
        k  : callable, kernel function k(X1, X2, x)
        x0 : np.ndarray, default hyperparameters
    """
    ep = 1e-16  # Smallest allowed amplitude

    if spec == "Gaussian":
        def k(X1, X2, x):
            return (ep + softplus(x[0])) * np.exp(-cdist(X1, X2) ** 2 / softplus(x[1])**2)
        x0 = np.array([1.0, 0.1])

    elif spec == "GaussianARD":
        def k(X1, X2, x):
            amp = ep + softplus(x[0])
            lengthscales = [cdist(X1[:, [i]], X2[:, [i]])**2 / softplus(x[i+1])**2 for i in range(d)]
            return amp * np.exp(-cellsum(lengthscales))
        x0 = np.concatenate(([1.0], 0.1 * np.ones(d)))

    elif spec == "white":
        def k(X1, X2, x):
            return (ep + softplus(x[0])) * white(X1, X2)
        x0 = np.array([1.0])

    elif spec == "Matern1/2":
        def k(X1, X2, x):
            return (ep + softplus(x[0])) * np.exp(-cdist(X1, X2) / softplus(x[1]))
        x0 = np.array([1.0, 1.0])

    elif spec == "Matern3/2":
        def k(X1, X2, x):
            r = cdist(X1, X2)
            l = softplus(x[1])
            sqrt3_r_l = np.sqrt(3) * r / l
            return (ep + softplus(x[0])) * (1 + sqrt3_r_l) * np.exp(-sqrt3_r_l)
        x0 = np.array([1.0, 1.0])

    elif isinstance(spec, tuple):  # GRE case: (B, base_kernel)
        B, k_name = spec
        def b(X):
            return np.sum(x2fx(X, B), axis=1)
        k_base, x0_base = kernel(k_name, d)
        p_base = len(x0_base)

        def k(X1, X2, x):
            amp = ep + softplus(x[0])
            return amp * b(X1)[:, None] * k_base(X1, X2, x[1:p_base+1]) * b(X2)[None, :]
        x0 = np.concatenate(([1.0], x0_base))

    else:
        raise ValueError(f"Unknown kernel specification: {spec}")

    return k, x0