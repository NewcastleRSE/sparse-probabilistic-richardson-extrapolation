##############################################################################
# Sparse Probabilistic Richardson Extrapolation (SPRE)
# Helper functions use by the SPRE methods.
# 
# Based on the methods and original MatLab code by Chris Oates.
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import jax.numpy as jnp

# Ensure 64-bit accuracy is used
from jax import config
config.update("jax_enable_x64", True)

def cellsum(arrays : list) -> jnp.array:
    """
    Pointwise addition for a collection of arrays stored in a list.

    Parameters:
        arrays : list of jnp.array
            List of arrays to sum element-wise.

    Returns:
        out : jnp.array
            Element-wise sum of all arrays.
    """

    out = arrays[0]
    for arr in arrays[1:]:
        out = jnp.add(out, arr)
    return out


def remove_row(arr : jnp.array, index : int) -> jnp.array:
    """
    Remove a row from a 2D NumPy array.

    Parameters:
        arr : jnp.array
            Input array.
        index : int
            Index of the row to remove.

    Returns:
        jnp.array
            Array with the specified row removed.
    """

    return jnp.delete(arr, index, axis=0)


def softplus(x : float) -> float:
    """
    Computes soft plus log(1 + exp(x)).

    Parameters:
        x : float
    Returns:
        float
    """

    return jnp.log1p(jnp.exp(x))  # log(1 + exp(x))

def stepwise(A : jnp.array, order : int) -> jnp.array:
    """
    Compute which high-order interactions to consider next.

    Parameters:
        A : jnp.array
            n_models x d array of current interactions.
        order : int
            current order to consider.

    Returns:
        jnp.array
            array of new interactions to consider.
    """
    n_models, d = A.shape
    
    # Mask for rows where the sum equals (order - 1)
    mask = A.sum(axis = 1) == (order - 1)
    A_filtered = A[mask]  # Shape: (k, d), where k is number of matching rows

    if A_filtered.shape[0] == 0:
        return jnp.empty((0, d), dtype=int)

    # For a selected row, add 1 to the first column, append row, add 1 to 2nd column, append row etc.
    # Giving d by d matrix. For each selected row this gives k, d by d matrices.
    eye_d = jnp.eye(d, dtype = int)  # Shape: (d, d)
    expanded = A_filtered[:, jnp.newaxis, :] + eye_d  # Shape: (k, d, d)

    # Reshape to 2D and remove duplicates
    out = jnp.unique(expanded.reshape(-1, d), axis=0)
    return out


def white(X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
    """
    White noise kernel (JIT-friendly, no Python loops or dynamic shapes).

    Parameters:
        X1 : jnp.ndarray of shape (n1, d)
        X2 : jnp.ndarray of shape (n2, d)

    Returns:
        jnp.ndarray of shape (n1, n2)
    """
    # Compare all pairs of rows between X1 and X2
    # X1[:, None, :] shape -> (n1, 1, d)
    # X2[None, :, :] shape -> (1, n2, d)
    # Broadcasting gives (n1, n2, d)
    eq = jnp.all(X1[:, None, :] == X2[None, :, :], axis=-1)

    # Convert boolean to float (1.0 for equal rows, 0.0 otherwise)
    return eq.astype(X1.dtype)

def x2fx(X : jnp.array, A : jnp.array) -> jnp.array:
    """
    Generate polynomial basis terms for each row in X using exponents in A.
    
    Parameters:
        X : jnp.array
            predictor matrix (n, d)
        A : (m, d) binary matrix (or integer exponents)
            powers to use for predictor variables

    Returns:
        jnp.array
            (n, m) design matrix where V[i,j] = prod_k X[i,k]^A[j,k]
    """
    
    return jnp.prod(jnp.array([X[:, [i]] ** A[:, i] for i in range(A.shape[1])]), axis=0)
