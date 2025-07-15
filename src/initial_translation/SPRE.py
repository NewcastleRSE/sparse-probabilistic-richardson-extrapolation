# Python modules
import jax.numpy as jnp
from jax import grad

# Application modules
from src.initial_translation.kernel import kernel
from src.initial_translation.helper_functions import x2fx, remove_row

def SPRE(A, X, Y, x, str_):
    """
    Sparse Probabilistic Richardson Extrapolation (SPRE).
    
    Parameters:
        A : np.ndarray of shape (m, d), binary matrix representing the sparse basis
        X : np.ndarray of shape (n_train, d), training inputs
        Y : np.ndarray of shape (n_train,), training outputs
        x : np.ndarray of shape (p,), kernel parameters
        str_ : str or tuple, kernel specification (c.f. function "kernel")
    
    Returns:
        out : dict with keys:
            - mu: scalar, predictive mean for f(0)
            - var: scalar, predictive variance for f(0)
            - mu_GP: function R^d -> R, predictive mean for fitted GP
            - cov_GP: function R^d x R^d -> R, predictive covariance for fitted GP
            - mu_cv: n_train x 1, LOOCV predictive means
            - var_cv: n_train x 1, LOOCV predictive variances
            - cv: scalar, LOOCV criterion
            - cv_grad: p x 1, gradient of LOOCV criterion
    """

    m, d = A.shape
    n_train = X.shape[0]
    p = len(x)

    # Data normalization
    ep = 1e-16
    nX = ep + (jnp.max(X, axis=0) - jnp.min(X, axis=0))
    nY = ep + (jnp.max(Y) - jnp.min(Y))
    Xn = X / nX
    Yn = Y / nY

    # Basis functions
    # A = m x d
    # X = n x d
    # Xs = n_test x d
    def V(A, X):
        return x2fx(X, A)  # n x m

    def v(A, Xs):
        return x2fx(Xs, A).T  # m x n_test

    # Kernel
    k_func = kernel(str_, d)
    def k(X1, X2, x): return k_func(X1, X2, x)

    # Residual term
    # A = m x d
    # X = n_train x d
    # Xs = n_test x d
    # x = p x 1
    def r(A, X, Xs, x):
        K_inv = jnp.linalg.inv(k(X, X, x))
        return v(A, Xs) - V(A, X).T @ K_inv @ k(X, Xs, x)

    # Coefficient estimator
    # A = m x d
    # X = n_train x d
    # Y = n_train x 1
    # x = p x 1
    def beta(A, X, Y, x):
        K_inv = jnp.linalg.inv(k(X, X, x))
        VA = V(A, X)
        return jnp.linalg.inv(VA.T @ K_inv @ VA) @ (VA.T @ K_inv @ Y)

    # Predictive mean
    # A = m x d
    # X = n_train x d
    # Y = n_train x 1
    # Xs = n_test x d
    # x = p x 1
    def mu_GP(A, X, Y, Xs, x):
        K_inv = jnp.linalg.inv(k(X, X, x))
        return k(Xs, X, x) @ K_inv @ Y + r(A, X, Xs, x).T @ beta(A, X, Y, x)

    # Predictive covariance
    # A = m x d
    # X = n_train x d
    # Xs = n_test x d
    # x = p x 1
    def cov_GP(A, X, Xs, x):
        K_inv = jnp.linalg.inv(k(X, X, x))
        VA = V(A, X)
        return (k(Xs, Xs, x)
                - k(Xs, X, x) @ K_inv @ k(X, Xs, x)
                + r(A, X, Xs, x).T @ jnp.linalg.inv(VA.T @ K_inv @ VA) @ r(A, X, Xs, x))

    # Cross-validation local loss (log-likelihood of test data)
    # A = m x d
    # X = n_train x d
    # Y = n_train x 1
    # Xs = n_test x d
    # Ys = n_test x 1
    # x = p x 1
    def cv_local_loss(A, X, Y, Xs, Ys, x):
        cov_val = cov_GP(A, X, Xs, x)
        mu_val = mu_GP(A, X, Y, Xs, x)
        diff = Ys - mu_val
        inv_cov = jnp.linalg.inv(cov_val)
        term1 = -0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * cov_val))
        term2 = -0.5 * diff.T @ inv_cov @ diff
        return term1 + term2

    # LOOCV loss 
    # A = m x d
    # X = n_train x d
    # Y = n_train x 1
    # x = p x 1
    def cv_loss(A, X, Y, x):
        return sum(
            cv_local_loss(
                A,
                remove_row(X, i),
                remove_row(Y, i),
                X[i:i+1, :],
                Y[i:i+1],
                x
            ) for i in range(n_train)
        )

    # LOOCV predictions
    mu_cv = jnp.array([
        nY * mu_GP(A, remove_row(Xn, i), remove_row(Yn, i), Xn[i:i+1, :], x)
        for i in range(n_train)
    ])

    var_cv = jnp.array([
        nY**2 * cov_GP(A, remove_row(Xn, i), Xn[i:i+1, :], x)
        for i in range(n_train)
    ]).flatten()

   
    # Define the function to calculate grdient from
    #def f_eval(x):
    #    return cv_loss(A, Xn, Yn, x)
    
    # Define the gradient function
    #gradient_function = grad(f_eval)

    # Evaluate gradient at x
    #gradient = gradient_function(x)

    # Output
    out = {
        "mu": nY * mu_GP(A, Xn, Yn, jnp.zeros((1, d)), x),
        "var": nY**2 * cov_GP(A, Xn, jnp.zeros((1, d)), x),
        "mu_GP": lambda Xs: nY * mu_GP(A, Xn, Yn, Xs / nX, x),
        "cov_GP": lambda Xs: nY**2 * cov_GP(A, Xn, Xs / nX, x),
        "mu_cv": mu_cv,
        "var_cv": var_cv,
        "cv": cv_loss(A, Xn, Yn, x),
        #"cv_grad": jnp.array(gradient)
    }

    return out
