# Python modules
#import numpy as np
import jax.numpy as jnp
from sklearn.neighbors import NearestNeighbors

# Application modules
from src.initial_translation.helper_functions import x2fx

def MRE(A, X, Y):
    """
    Multivariate Richardson Extrapolation.

    Parameters:
        A : np.ndarray
            of shape (m, d). Sparse basis (binary matrix).
        X : np.ndarray
            of shape (n_train, d). Training inputs.
        Y : np.ndarray
            of shape (n_train,) or (n_train, 1). Training outputs.

    Returns:
        float
           'mu', the predicted f(0)
    """
    m, d = A.shape

    # Ensure Y is a flat 1D array
    Y = Y.flatten()

    # Find m nearest neighbors to 0 in X
    nbrs = NearestNeighbors(n_neighbors=m).fit(X)
    indices = nbrs.kneighbors(jnp.zeros((1, d)), return_distance = False)
    idx = indices.flatten()

    X_sel = X[idx]
    Y_sel = Y[idx]

    # Normalization
    ep = 1e-16 # smallest permitted normalising constant
    nX = ep + jnp.ptp(X_sel, axis=0)  # normalising constant for X
    nY = ep + jnp.ptp(Y_sel)          # normalising constant for Y

    Xn = X_sel / nX
    Yn = Y_sel / nY
   
    # Polynomial fit
    V = x2fx(Xn, A)  # Design matrix
    eval = x2fx(jnp.zeros((1, d)), A)  # Evaluation point

    # Least-squares solution
    coeffs = jnp.linalg.lstsq(V, Yn, rcond=None)[0]
    mu = nY * eval @ coeffs

    return mu
