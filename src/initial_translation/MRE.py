import numpy as np
#from sklearn.neighbors import NearestNeighbors

def x2fx(X : np.ndarray, A : np.ndarray) -> np.ndarray:
    """
    Generate polynomial basis terms for each row in X using exponents in A.
    
    Parameters:
        X : np.ndarray
            predictor matrix (n, d)
        A : (m, d) binary matrix (or integer exponents)
            powers to use for predictor variables

    Returns:
        np.ndarray
            (n, m) design matrix where V[i,j] = prod_k X[i,k]^A[j,k]
    """
    return np.prod([X[:, [i]] ** A[:, i] for i in range(A.shape[1])], axis=0)

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
        dict
            with key 'mu', the predicted f(0)
    """
    m, d = A.shape

    # Ensure Y is a flat 1D array
    Y = Y.flatten()

    # Find m nearest neighbors to 0 in X
    nbrs = NearestNeighbors(n_neighbors=m).fit(X)
    distances, indices = nbrs.kneighbors(np.zeros((1, d)))
    idx = indices.flatten()

    X_sel = X[idx]
    Y_sel = Y[idx]

    # Normalization
    ep = 1e-16
    nX = ep + np.ptp(X_sel, axis=0)  # range along each column
    nY = ep + np.ptp(Y_sel)

    Xn = X_sel / nX
    Yn = Y_sel / nY

    # Polynomial fit
    V = x2fx(Xn, A)  # Design matrix
    v = x2fx(np.zeros((1, d)), A)  # Evaluation point at 0

    # Least-squares solution
    coeffs = np.linalg.lstsq(V, Yn, rcond=None)[0]
    mu = nY * v @ coeffs

    return mu #{"mu": float(mu)}
