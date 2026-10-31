##############################################################################
# Toy illustration of sequential experimental design for SPRE
# (Figure 4 and Appendix C.2 of "Sparse Probabilistic Richardson Extrapolation").
#
# Python port of matlab_code/experiment_3.m, with the computational budget now
# applied to the total cost of each new batch of simulations, as described in
# the paper.
#
# Usage, from the root directory:
#   python ./src/run_toy_design.py
#
# Results are written to data/toy_design/results/toy_design.json.
##############################################################################

import json
from pathlib import Path
import numpy as np

from run_sequential_design import fit_spre, select_batch, _softplus

RESULTS = Path("data/toy_design/results")

D = 2                  # dimension
BUDGET = 2.0           # computational budget C for each iteration
N_ITERATIONS = 8       # iterations of sequential design
N_ATTEMPTS = 1000      # candidate batches examined per iteration
KERNEL = "Matern1/2"   # kernel used to learn (k, A) and for experimental design


def make_simulator(seed: int = 0, n_grid: int = 2001):
    """f(x) = 1 + x_1 - 2 x_2 + 3 x_1^2 + 0.0001 x_1 x_2 eps(x), with eps a realised white noise process."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_grid, n_grid))

    def eps(X):  # bilinear interpolation of i.i.d. standard normals on a fine grid of [0,1]^2
        u = np.clip(X, 0, 1) * (n_grid - 1)
        i0 = np.minimum(np.floor(u).astype(int), n_grid - 2)
        w = u - i0
        a = noise[i0[:, 0], i0[:, 1]]
        b = noise[i0[:, 0] + 1, i0[:, 1]]
        c = noise[i0[:, 0], i0[:, 1] + 1]
        e = noise[i0[:, 0] + 1, i0[:, 1] + 1]
        return (1 - w[:, 0]) * (1 - w[:, 1]) * a + w[:, 0] * (1 - w[:, 1]) * b \
            + (1 - w[:, 0]) * w[:, 1] * c + w[:, 0] * w[:, 1] * e

    return lambda X: 1 + X[:, 0] - 2 * X[:, 1] + 3 * X[:, 0] ** 2 + 0.0001 * X[:, 0] * X[:, 1] * eps(X)


def cost(X):
    """c(x) = (x_1 x_2)^{-1}."""
    X = np.atleast_2d(X)
    return 1.0 / (X[:, 0] * X[:, 1])


def main(seed: int = 0) -> None:
    f = make_simulator(seed)
    rng = np.random.default_rng(seed)
    X = 1.0 / np.array([[1, 1], [2, 1], [1, 2]], dtype=float)
    X = X - 0.01 * rng.uniform(size=X.shape)   # small jitter, as in the original experiment
    Y = f(X)
    iterations = []
    for it in range(1, N_ITERATIONS + 1):
        out = fit_spre(X, Y, KERNEL, max_order=20)
        A = out["A"]
        sigma = float(np.sqrt(1e-16 + _softplus(out["params"][0])))
        batch, spent = select_batch(rng, KERNEL, out["params"], A, X, cost, BUDGET, 1.0, 0.0, N_ATTEMPTS)
        iterations.append({"iteration": it, "X": X.tolist(), "X_new": batch.tolist(), "cost_new": spent,
                           "A": A.tolist(), "sigma": sigma, "estimate": out["mu"], "sd": out["sd"]})
        print(f"iteration {it}: A = {A.tolist()}, sigma = {sigma:.2g}, new points = {len(batch)}, "
              f"estimate = {out['mu']:.6f} +/- {out['sd']:.2g}", flush=True)
        X = np.vstack([X, batch])
        Y = np.concatenate([Y, f(batch)])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "toy_design.json").write_text(json.dumps({"budget": BUDGET, "kernel": KERNEL,
                                                         "iterations": iterations}, indent=1))


if __name__ == "__main__":
    main()
