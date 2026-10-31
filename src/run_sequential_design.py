##############################################################################
# Sequential (cost-constrained) experimental design for SPRE, compared with
# a fixed (one-shot) design, as a function of computational cost.
#
# Usage, from the root directory:
#   python ./src/run_sequential_design.py ./data/mujoco/input_two_spheres_design.json [fixed|sequential|plot|all]
#
# The parameter file contains the usual model settings (see README.md) together
# with a "design" dictionary controlling the experiment:
#
#   n_processes            number of model simulations to run in parallel
#   max_order              maximum order of the index set A in stepwise selection
#   cost                   nominal cost model; "inverse_time_step" sets c(x) = 1 / x_1
#   h_values_fixed         scalings h for the fixed design X_n^h = {h x_i : x_i in X}
#   h_values_sequential    scalings h for which a sequential design is also run
#   initial_rows           rows of X which, scaled by h, initialise the sequential
#                          design; these points are common to both designs
#   n_iterations           number of iterations of sequential design
#   candidate_u_min        candidate points are h * u with u ~ Uniform([candidate_u_min, 1]^d)
#   n_attempts             number of random candidate batches examined per iteration
#   seeds                  random seeds, one sequential design is run for each seed and h
#
# For each h, the sequential design is constrained to the same region [0, h]^d and
# the same total computational budget sum_i c(h x_i) as the fixed design X_n^h.
# After the initial points have been simulated, the remaining budget is divided
# equally among n_iterations iterations, each of which alternates between
# (i) learning the index set A (and kernel parameters) with the stepwise LOOCV
# procedure of SPRE, and (ii) selecting a batch of new simulations that minimises
# the posterior variance of f(0) subject to the per-iteration budget, by stochastic
# search over randomly generated batches. The posterior variance of f(0) does not
# depend on the (as yet unobserved) simulator outputs at the new design points.
#
# Written for the revision of "Sparse Probabilistic Richardson Extrapolation".
##############################################################################

# Python modules
import sys
import json
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# Application modules
from models.models_utils import get_model
from sparse_pre.extrapolation import extrapolation


##############################################################################
# Parallel model simulation (each worker holds its own model object)
##############################################################################

_worker_model = None

def _init_worker(parameter_filename: str) -> None:
    global _worker_model
    _worker_model = get_model(parameter_filename, skip_true_value_calc=True)

def _simulate_one(x) -> float:
    return float(_worker_model.run_model(np.asarray(x, dtype=float)))

def simulate(parameter_filename: str, X: np.ndarray, n_processes: int) -> np.ndarray:
    """Run (or look up in the cache) the simulator at each row of X."""
    if n_processes <= 1:
        _init_worker(parameter_filename)
        return np.array([_simulate_one(x) for x in X])
    with Pool(n_processes, _init_worker, (parameter_filename,)) as pool:
        return np.array(pool.map(_simulate_one, list(X)))


##############################################################################
# Cost model, SPRE fit and design criterion
##############################################################################

def make_cost(name: str):
    if name == "inverse_time_step":
        # Nominal cost proportional to the number of time steps, c(x) = 1 / x_1
        return lambda X: 1.0 / np.atleast_2d(X)[:, 0]
    raise ValueError(f"Unknown cost model: {name}")


def fit_spre(X: np.ndarray, Y: np.ndarray, kernel: str, max_order: int) -> dict:
    options = {"name": "SPRE", "k_name": kernel, "plot": False, "max_order": max_order}
    out = extrapolation(X, Y, options)
    return {
        "mu": float(np.asarray(out["mu"]).ravel()[0]),
        "sd": float(np.sqrt(np.asarray(out["var"]).ravel()[0])),
        "A": np.asarray(out["A"], dtype=int),
        "params": np.asarray(out["x"], dtype=float),
    }


def _softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def _kernel(name, params, X1, X2, ep=1e-16):
    """Numpy versions of the kernels in sparse_pre.SPRE (inputs already normalised)."""
    D = np.sqrt(np.maximum(((X1[:, None, :] - X2[None, :, :]) ** 2).sum(-1), 0.0))
    amp = ep + _softplus(params[0])
    if name == "white":
        return amp * (D == 0).astype(float)
    if name == "Matern1/2":
        return amp * np.exp(-D / _softplus(params[1]))
    if name == "Matern3/2":
        r = np.sqrt(3) * D / _softplus(params[1])
        return amp * (1 + r) * np.exp(-r)
    if name == "Gaussian":
        return amp * np.exp(-D ** 2 / _softplus(params[1]) ** 2)
    raise ValueError(f"Design criterion not implemented for kernel {name}")


def posterior_variance_at_zero(kernel, params, A, X, nX) -> float:
    """
    Posterior variance of f(0) under SPRE, up to the normalisation of the outputs, for the
    design X (which does not depend on the simulator outputs). Inputs are normalised by nX,
    as in sparse_pre.SPRE. Returns np.inf if X is not P_A-unisolvent.
    """
    Xn = X / nX
    V = np.stack([np.prod(Xn ** a, axis=1) for a in A], axis=1)
    if V.shape[0] < V.shape[1] or np.linalg.matrix_rank(V) < V.shape[1]:
        return np.inf
    if len(np.unique(Xn, axis=0)) < len(Xn):
        return np.inf
    zero = np.zeros((1, X.shape[1]))
    K = _kernel(kernel, params, Xn, Xn)
    k0 = _kernel(kernel, params, Xn, zero)
    v0 = np.stack([np.prod(zero ** a, axis=1) for a in A], axis=1).T
    Kinv = np.linalg.inv(K)
    r = v0 - V.T @ Kinv @ k0
    M = np.linalg.inv(V.T @ Kinv @ V)
    var = _kernel(kernel, params, zero, zero) - k0.T @ Kinv @ k0 + r.T @ M @ r
    return float(var[0, 0])


def random_batch(rng, cost, budget, h, u_min, d, max_tries=1000):
    """Sample candidate points h * u sequentially, stopping before the budget is exceeded."""
    batch, spent = [], 0.0
    for _ in range(max_tries):
        x = h * rng.uniform(u_min, 1.0, size=d)
        c = float(cost(x)[0])
        if spent + c <= budget:
            batch.append(x)
            spent += c
        elif batch:
            break
    return np.array(batch), spent


def select_batch(rng, kernel, params, A, X, cost, budget, h, u_min, n_attempts):
    nX = np.max(X, axis=0) - np.min(X, axis=0)  # normalisation used by SPRE for the current data
    best, best_val, best_spent = None, np.inf, 0.0
    for _ in range(n_attempts):
        batch, spent = random_batch(rng, cost, budget, h, u_min, X.shape[1])
        if len(batch) == 0:
            continue
        val = posterior_variance_at_zero(kernel, params, A, np.vstack([X, batch]), nX)
        if val < best_val:
            best, best_val, best_spent = batch, val, spent
    return best, best_spent


##############################################################################
# Experiments
##############################################################################

def run_fixed(model, parameter_filename, design, results_dir, cost):
    X_ref = np.array(model.X, dtype=float)
    rows = []
    for h in design["h_values_fixed"]:
        X = h * X_ref
        Y = simulate(parameter_filename, X, design["n_processes"])
        out = fit_spre(X, Y, model.extrapolation_kernel, design["max_order"])
        rows.append({
            "h": h,
            "cost": float(np.sum(cost(X))),
            "n": len(Y),
            "mu": out["mu"],
            "sd": out["sd"],
            "abs_err": abs(out["mu"] - model.true_value),
            "min_raw_abs_err": float(np.min(np.abs(Y - model.true_value))),
            "A": ";".join("".join(map(str, a)) for a in out["A"]),
        })
        print(f"[fixed] h = {h:.3e}, cost = {rows[-1]['cost']:.3e}, abs err = {rows[-1]['abs_err']:.3e}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / design["results_fixed_filename"], sep="\t", index=False)
    return df


def run_sequential(model, parameter_filename, design, results_dir, cost):
    X_ref = np.array(model.X, dtype=float)
    kernel = model.extrapolation_kernel
    rows = []
    for h in design["h_values_sequential"]:
        total_budget = float(np.sum(cost(h * X_ref)))  # cost of the fixed design X_n^h
        for seed in design["seeds"]:
            rng = np.random.default_rng(seed)
            X = h * X_ref[design["initial_rows"]]
            Y = simulate(parameter_filename, X, design["n_processes"])
            spent_total = float(np.sum(cost(X)))
            budget = (total_budget - spent_total) / design["n_iterations"]
            for it in range(design["n_iterations"] + 1):
                out = fit_spre(X, Y, kernel, design["max_order"])
                rows.append({
                    "h": h,
                    "seed": seed,
                    "iteration": it,
                    "cost": spent_total,
                    "n": len(Y),
                    "mu": out["mu"],
                    "sd": out["sd"],
                    "abs_err": abs(out["mu"] - model.true_value),
                    "min_raw_abs_err": float(np.min(np.abs(Y - model.true_value))),
                    "A": ";".join("".join(map(str, a)) for a in out["A"]),
                })
                print(f"[sequential] h = {h:.3e}, seed {seed}, iteration {it}, n = {len(Y)}, "
                      f"cost = {spent_total:.3e}, abs err = {rows[-1]['abs_err']:.3e}, A = {rows[-1]['A']}", flush=True)
                if it == design["n_iterations"]:
                    break
                batch, spent = select_batch(rng, kernel, out["params"], out["A"], X, cost, budget, h,
                                            design["candidate_u_min"], design["n_attempts"])
                if batch is None:
                    break
                X = np.vstack([X, batch])
                Y = np.concatenate([Y, simulate(parameter_filename, batch, design["n_processes"])])
                spent_total += spent
            # Save the final design
            np.savetxt(results_dir / design["results_sequential_filename"].replace(".dat", f"_design_h{h:.3e}_seed{seed}.dat"),
                       np.column_stack([X, Y]), delimiter="\t", fmt="%.17g",
                       header="\t".join([f"X{j+1}" for j in range(X.shape[1])] + ["Y"]), comments="")
            pd.DataFrame(rows).to_csv(results_dir / design["results_sequential_filename"], sep="\t", index=False)
    return pd.DataFrame(rows)


def plot(design, results_dir):
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 14,
        "font.size": 14,
        "legend.fontsize": 12,
        "savefig.dpi": 300
    })
    fixed = pd.read_csv(results_dir / design["results_fixed_filename"], sep="\t")
    seq = pd.read_csv(results_dir / design["results_sequential_filename"], sep="\t")
    final = seq[seq["iteration"] == seq["iteration"].max()]
    summary = final.groupby("h").agg(cost=("cost", "median"), err=("abs_err", "median"))

    plt.close("all")
    plt.figure(figsize=(7, 6))
    plt.plot(fixed["cost"], fixed["min_raw_abs_err"], marker="o", markersize=4, linewidth=2, alpha=0.4,
             color="tab:gray", zorder=0, label="most accurate raw estimate (fixed design)")
    plt.plot(fixed["cost"], fixed["abs_err"], marker="s", color="k", label="SPRE (fixed design)")
    plt.plot(final["cost"], final["abs_err"], linestyle="none", marker="^", markersize=6, alpha=0.5,
             color="tab:blue", label="SPRE (sequential design), each seed")
    plt.plot(summary["cost"], summary["err"], marker="^", color="tab:blue", label="SPRE (sequential design), median")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"total computational cost $\sum_i c(\mathbf{x}_i)$")
    plt.ylabel("absolute error")
    plt.title(design.get("plot_title", ""))
    plt.grid(True)
    plt.minorticks_off()
    plt.grid(True, which="major", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        plt.savefig(results_dir / design["results_plot_filename"].replace(".png", ext))
    print(f"Saved {results_dir / design['results_plot_filename']}")


if __name__ == "__main__":
    parameter_filename = sys.argv[1]
    part = sys.argv[2] if len(sys.argv) > 2 else "all"
    with open(parameter_filename) as f:
        design = json.load(f)["design"]
    results_dir = Path(parameter_filename).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cost = make_cost(design["cost"])
    if part in ("fixed", "sequential", "all"):
        model = get_model(parameter_filename)  # computes the true value f(x_0)
        print(f"True value: {model.true_value!r}", flush=True)
    t0 = time.perf_counter()
    if part in ("fixed", "all"):
        run_fixed(model, parameter_filename, design, results_dir, cost)
    if part in ("sequential", "all"):
        run_sequential(model, parameter_filename, design, results_dir, cost)
    if part in ("plot", "all"):
        plot(design, results_dir)
    print(f"Total time: {time.perf_counter() - t0:.1f} s")
