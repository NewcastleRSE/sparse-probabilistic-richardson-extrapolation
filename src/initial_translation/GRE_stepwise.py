# Python modules
import jax.numpy as jnp
from tqdm import tqdm  # For progress bars

# Application modules
from kernel import kernel
from helper_functions import stepwise
from SPRE_opt import SPRE_opt
from SPRE import SPRE

def GRE_stepwise(X, Y, k_name):
    """
    Stepwise model selection for GRE (Gauss-Richardson Extrapolation).

    Parameters:
        X : jnp.ndarray of shape (n_train, d)
            Training inputs.
        Y : jnp.ndarray of shape (n_train,)
            Training outputs.
        k_name : str
            Kernel name: "Gaussian", "GaussianARD", "Matern1/2", "Matern3/2", or "white".

    Returns:
        out : dict
            Dictionary with predictive mean, variance, and fitted model details.
    """
    d = X.shape[1]  # data dimension

    A = jnp.array([[0, 0]])  # mean function: just the intercept
    B = jnp.zeros((1, d))    # initial rate function: only intercept
    order = 0

    fit = SPRE_opt(A, X, Y, (B, k_name))
    cv = fit["cv"]

    carry_on = True
    while carry_on:
        order += 1
        B_extra = stepwise(B, order)  # Generate all predictors of the next order
        n_extra = B_extra.shape[0]

        print(f"Fitting interactions of order {order}...")

        to_include = jnp.zeros(n_extra, dtype=bool)
        for i in range(n_extra):
            B_new = jnp.vstack([B, B_extra[i, :]])
            fit_new = SPRE_opt(A, X, Y, (B_new, k_name))
            cv_new = fit_new["cv"]

            if cv_new < cv:
                to_include[i] = True

            # Progress bar substitute
            #cwbar((i + 1) / n_extra)

        if jnp.any(to_include):
            B_updated = jnp.vstack([B, B_extra[to_include, :]])
            fit_updated = SPRE_opt(A, X, Y, (B_updated, k_name))
            cv_updated = fit_updated["cv"]

            if cv_updated >= cv:
                carry_on = False
            else:
                fit = fit_updated
                B = B_updated
                cv = cv_updated
        else:
            carry_on = False

        #cwbar("done")

    # Final GP fit with optimal parameters
    x_opt = fit["x"]
    out = SPRE(A, X, Y, x_opt, (B, k_name))

    return out
