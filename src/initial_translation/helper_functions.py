# Python modules
# import numpy as np
import jax.numpy as jnp

def cellsum(arrays : list) -> jnp.ndarray:
    """
    Pointwise addition for a collection of arrays stored in a list.

    Parameters:
        arrays : list of jnp.ndarray
            List of arrays to sum element-wise.

    Returns:
        out : jnp.ndarray
            Element-wise sum of all arrays.
    """
    out = arrays[0]
    for arr in arrays[1:]:
        out = jnp.add(out, arr)
    return out

def remove_row(arr : jnp.ndarray, index : int) -> jnp.ndarray:
    """
    Remove a row from a 2D NumPy array.

    Parameters:
        arr : jnp.ndarray
            Input array.
        index : int
            Index of the row to remove.

    Returns:
        jnp.ndarray
            Array with the specified row removed.
    """
    return jnp.delete(arr, index, axis=0)

def softplus(x : float):
    return jnp.log1p(jnp.exp(x))  # log(1 + exp(x))

def stepwise(A : jnp.ndarray, order : int) -> jnp.ndarray:
    """
    Compute which high-order interactions to consider next.

    Parameters:
        A : jnp.ndarray
            n_models x d array of current interactions.
        order : int
            current order to consider.

    Returns:
        jnp.ndarray
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

def white(X1, X2):
    """
    White noise kernel.

    Parameters:
        X1 : jnp.ndarray
            of shape (n1, d)
        X2 : jnp.ndarray
            of shape (n2, d)

    Returns:
        jnp.ndarray
            of shape (n1, n2)
    """
    n1 = X1.shape[0]
    n2 = X2.shape[0]

    out = jnp.zeros((n1, n2))

    # Loop thro' each row in X1 and check where it appears in X2 if anywhere
    # out[i, j] = 1 indicates that row i in X1 is the same as row j in X2
    for row in range(n1):
        matching_rows = jnp.where((X2 == X1[row,:]).all(axis = 1))
        out = out.at[row, matching_rows].set(1)

    return out

def x2fx(X : jnp.ndarray, A : jnp.ndarray) -> jnp.ndarray:
    """
    Generate polynomial basis terms for each row in X using exponents in A.
    
    Parameters:
        X : jnp.ndarray
            predictor matrix (n, d)
        A : (m, d) binary matrix (or integer exponents)
            powers to use for predictor variables

    Returns:
        jnp.ndarray
            (n, m) design matrix where V[i,j] = prod_k X[i,k]^A[j,k]
    """
    return jnp.prod(jnp.array([X[:, [i]] ** A[:, i] for i in range(A.shape[1])]), axis=0)