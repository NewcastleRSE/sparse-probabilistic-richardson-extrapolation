import numpy as np

def cellsum(arrays):
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

def remove_row(arr, index):
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

def softplus(x):
    return np.log1p(np.exp(x))  # log(1 + exp(x)) with better numerical stability

def stepwise(A, order):
    """
    Compute which high-order interactions to consider next.

    Parameters:
        A (np.ndarray): n_models x d array of current interactions.
        order (int): current order to consider.

    Returns:
        np.ndarray: array of new interactions to consider.
    """
    out = []
    n_models, d = A.shape
    for i in range(n_models):
        Ai = A[i, :]
        if Ai.sum() == (order - 1):  # ignore interactions of order-2 and lower
            for j in range(d):
                Aij = Ai.copy()
                Aij[j] += 1  # increment the j-th entry
                out.append(Aij)
    if len(out) == 0:
        return np.empty((0, d), dtype=int)
    out = np.unique(np.array(out), axis=0)
    return out

def white(X1, X2):
    """
    White noise kernel.

    Parameters:
        X1: np.ndarray of shape (n1, d)
        X2: np.ndarray of shape (n2, d)

    Returns:
        out: np.ndarray of shape (n1, n2)
    """
    n1 = X1.shape[0]
    n2 = X2.shape[0]

    out = np.zeros((n1, n2))

    # Find matching rows of X1 in X2, return indices or -1 if not found
    # This mimics MATLAB's ismember with 'rows'
    # We can use a structured array view for row-wise comparison

    dtype = np.dtype((np.void, X1.dtype.itemsize * X1.shape[1]))
    X1_view = X1.view(dtype).ravel()
    X2_view = X2.view(dtype).ravel()

    # For each row in X1, find index in X2 or -1 if not found
    Locb = np.array([np.where(X2_view == row)[0][0] if np.any(X2_view == row) else -1 for row in X1_view])

    for i in range(n1):
        if Locb[i] != -1:
            out[i, Locb[i]] = 1

    return out
