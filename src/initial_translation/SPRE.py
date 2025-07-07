import numpy as np

def SPRE(A, X, Y, x, str_):
    """
    Sparse Probabilistic Richardson Extrapolation (SPRE).
    
    Parameters:
        A : np.ndarray of shape (m, d), binary matrix for sparse basis
        X : np.ndarray of shape (n_train, d), training inputs
        Y : np.ndarray of shape (n_train,), training outputs
        x : np.ndarray of shape (p,), kernel parameters
        str_ : str or tuple, kernel specification or (B, kernel_name) tuple
    
    Returns:
        out : dict with keys:
            - mu: predictive mean at 0
            - var: predictive variance at 0
            - mu_GP: predictive mean function
            - cov_GP: predictive covariance function
            - mu_cv: LOOCV predictive means
            - var_cv: LOOCV predictive variances
            - cv: LOOCV criterion
            - cv_grad: gradient of LOOCV criterion
    """

    m, d = A.shape
    n_train = X.shape[0]
    p = len(x)

    # Data normalization
    ep = 1e-16
    nX = ep + (np.max(X, axis=0) - np.min(X, axis=0))
    nY = ep + (np.max(Y) - np.min(Y))
    Xn = X / nX
    Yn = Y / nY

    # Basis functions
    def V(A, X):
        return x2fx(X, A)  # n x m

    def v(A, Xs):
        return x2fx(Xs, A).T  # m x n_test

    # Kernel
    k_func, _ = kernel(str_, d)
    def k(X1, X2, x): return k_func(X1, X2, x)

    # Residual term
    def r(A, X, Xs, x):
        K_inv = np.linalg.inv(k(X, X, x))
        return v(A, Xs) - V(A, X).T @ K_inv @ k(X, Xs, x)

    # Coefficient estimator
