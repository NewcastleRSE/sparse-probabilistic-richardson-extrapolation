import numpy as np

def cellsum(arrays : list) -> np.ndarray:
    """
    Pointwise addition for a collection of arrays stored in a list.

    Parameters:
        arrays : list of np.ndarray
            List of arrays to sum element-wise.

    Returns:
        out : np.ndarray
            Element-wise sum of all arrays.
    """
    out = arrays[0]
    for arr in arrays[1:]:
        out = np.add(out, arr)
    return out

def remove_row(arr : np.ndarray, index : int) -> np.ndarray:
    """
    Remove a row from a 2D NumPy array.

    Parameters:
        arr : np.ndarray
            Input array.
        index : int
            Index of the row to remove.

    Returns:
        np.ndarray
            Array with the specified row removed.
    """
    return np.delete(arr, index, axis=0)

def softplus(x : float):
    return np.log1p(np.exp(x))  # log(1 + exp(x))

def stepwise(A : np.ndarray, order : int) -> np.ndarray:
    """
    Compute which high-order interactions to consider next.

    Parameters:
        A : np.ndarray
            n_models x d array of current interactions.
        order : int
            current order to consider.

    Returns:
        np.ndarray
            array of new interactions to consider.
    """
    n_models, d = A.shape
    
    # Mask for rows where the sum equals (order - 1)
    mask = A.sum(axis = 1) == (order - 1)
    A_filtered = A[mask]  # Shape: (k, d), where k is number of matching rows

    if A_filtered.shape[0] == 0:
        return np.empty((0, d), dtype=int)

    # For a selected row, add 1 to the first column, append row, add 1 to 2nd column, append row etc.
    # Giving d by d matrix. For each selected row this gives k, d by d matrices.
    eye_d = np.eye(d, dtype = int)  # Shape: (d, d)
    expanded = A_filtered[:, np.newaxis, :] + eye_d  # Shape: (k, d, d)

    # Reshape to 2D and remove duplicates
    out = np.unique(expanded.reshape(-1, d), axis=0)
    return out

def white(X1, X2):
    """
    White noise kernel.

    Parameters:
        X1 : np.ndarray
            of shape (n1, d)
        X2 : np.ndarray
            of shape (n2, d)

    Returns:
        np.ndarray
            of shape (n1, n2)
    """
    n1 = X1.shape[0]
    n2 = X2.shape[0]

    out = np.zeros((n1, n2))

    # Loop thro' each row in X1 and check where it appears in X2 if anywhere
    # out[i, j] = 1 indicates that row i in X1 is the same as row j in X2
    for row in range(n1):
        matching_rows = np.where((X2 == X1[row,:]).all(axis = 1))
        out[row, matching_rows] = 1

    return out
