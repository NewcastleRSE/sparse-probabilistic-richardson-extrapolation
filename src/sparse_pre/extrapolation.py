# Python modules
import jax.numpy as jnp
from jax import grad, debug
import matplotlib.pyplot as plt

# Application modules
from SPRE import SPRE

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

    # Set up SPRE object
    if name == "SPRE":
        spre = SPRE(k_name, X.shape[1])
    elif name == "GRE":
        spre = SPRE(k_name, X.shape[1], jnp.zeros((1, X.shape[1]), dtype=int))
    elif name == "MRE":
        raise NotImplementedError("MRE extrapolation is not implemented yet.")
    else:
        raise ValueError(f"Unknown extrapolation method: {name}")
    
    # Set data
    spre.set_normalised_data(X, Y)

    # Select extrapolation method
    if name == "SPRE" or name == "GRE":
        out = spre.stepwise_selection()
    elif name == "MRE":
        raise NotImplementedError("MRE extrapolation is not implemented yet.")
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