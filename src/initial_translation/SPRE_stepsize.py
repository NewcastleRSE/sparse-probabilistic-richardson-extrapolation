# Python modules
import numpy as np
from tqdm import tqdm  # For progress bars

# Application modules
from src.initial_translation.helper_functions import stepwise
from src.initial_translation.SPRE_opt import SPRE_opt
from src.initial_translation.SPRE import SPRE

def SPRE_stepwise(X, Y, k_name):
    """
    Stepwise model selection for SPRE.

    Parameters:
        X       : ndarray of shape (n_train, d), training inputs
        Y       : ndarray of shape (n_train,), training outputs
        k_name  : str, kernel name ("Gaussian", "GaussianARD", "Matern1/2", "Matern3/2", "white")

    Returns:
        out     : dict, result of SPRE using optimal model
    """

    d = X.shape[1]
    A = np.zeros((1, d), dtype=int)  # Initialize with intercept
    order = 0
    fit = SPRE_opt(A, X, Y, k_name)
    cv = fit['cv']

    carry_on = True

    while carry_on:
        order += 1
        A_extra = stepwise(A, order)  # Generate new predictors of given order
        n_extra = A_extra.shape[0]
        to_include = np.zeros(n_extra, dtype=bool)

        print(f"Fitting interactions of order {order}:")

        for i in tqdm(range(n_extra), desc="Stepwise progress"):
            A_new = np.vstack([A, A_extra[i]])
            fit_new = SPRE_opt(A_new, X, Y, k_name)
            cv_new = fit_new['cv']
            if cv_new < cv:
                to_include[i] = True

        if np.any(to_include):
            A_updated = np.vstack([A, A_extra[to_include]])
            fit_updated = SPRE_opt(A_updated, X, Y, k_name)
            cv_updated = fit_updated['cv']
            if cv_updated >= cv:
                carry_on = False
            else:
                A = A_updated
                fit = fit_updated
                cv = cv_updated
        else:
            carry_on = False

    # Optimal parameters
    x_opt = fit['x']

    # Final model with best kernel parameters and basis A
    out = SPRE(A, X, Y, x_opt, k_name)

    return out
