# Python modules
import jax.numpy as jnp
from jax import grad, debug
import matplotlib.pyplot as plt

# Application modules
from SPRE_stepwise import SPRE_stepwise
from GRE_stepwise import GRE_stepwise

def extrapolation(X, Y, options=None):
    """
    Extrapolation to estimate f(0) from input-output training data (X, Y).

    Parameters:
        X : np.ndarray of shape (n_train, d)
            Training input vectors.
        Y : np.ndarray of shape (n_train,)
            Training scalar outputs.
        options : dict, optional
            Extrapolation options with keys:
                - "name"   : str, one of {"MRE", "GRE", "SPRE"} (default: "SPRE")
                - "k_name" : str, one of {"Gaussian", "GaussianARD", "Matern1/2", "Matern3/2", "white"} (default: "white")
                - "plot"   : bool, whether to plot LOOCV results (default: True)

    Returns:
        out : dict
            A dictionary containing:
                - "mu"     : predictive mean for f(0)
                - "var"    : predictive variance (if available)
                - "mu_cv"  : LOOCV means (optional)
                - "var_cv" : LOOCV variances (optional)
    """

    # Default options
    if options is None:
        options = {}
    name = options.get("name", "SPRE")
    k_name = options.get("k_name", "white")
    plot = options.get("plot", True)

    # Select extrapolation method
    if name == "MRE":
        raise NotImplementedError("MRE extrapolation is not implemented yet.")
    elif name == "GRE":
        out = GRE_stepwise(X, Y, k_name)
    elif name == "SPRE":
        out = SPRE_stepwise(X, Y, k_name)
    else:
        raise ValueError(f"Unknown extrapolation method: {name}")

    errors = jnp.sqrt(out["var_cv"])

    # Plot LOOCV fit if applicable
    if plot and name != "MRE" and "mu_cv" in out and "var_cv" in out:
        n_train = X.shape[0]
        plt.figure()
        plt.errorbar(jnp.arange(1, n_train + 1), out["mu_cv"].flatten(), yerr = errors, fmt='bo', label='predicted')
        plt.scatter(jnp.arange(1, n_train + 1), Y, color='k', marker='x', label='actual')
        plt.xticks(jnp.arange(1, n_train + 1))
        plt.xlabel(r'$i$')
        plt.ylabel(r'$f(\mathbf{x}_i)$')
        plt.title(f"Leave-one-out cross validation ({name})")
        plt.legend()
        plt.show()

    return out