##############################################################################
# Cubature experiments for the paper "Sparse Probabilistic Richardson
# Extrapolation" (Section 3 and Appendix C.1).
#
# A Riemann midpoint approximation to the integral of an integrand g on [0,1]^d
# is extrapolated with respect to the bin widths x = (x_1, ..., x_d), using MRE,
# GRE and SPRE. Python port of the original MATLAB scripts experiment_1.m and
# experiment_2.m (see matlab_code/).
#
# Usage, from the root directory:
#   python ./src/run_cubature_experiments.py [illustration|convergence|all]
#
# Results are written to data/cubature/results/.
##############################################################################

import sys
import json
import itertools
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import sympy as sp

from jax import config
config.update("jax_enable_x64", True)

from sparse_pre.SPRE import SPRE
import sparse_pre.MRE as MRE

RESULTS = Path("data/cubature/results")
EPS = np.finfo(float).eps


##############################################################################
# Integrand, Riemann sums and the exact integral
##############################################################################

def integrand(d: int, s: int):
    """
    g(t) = 1 + phi(sum(t) / d), where phi is |z - 1/2| integrated 2s + 2 times
    (so that g is in C^{2s+2}) and normalised so that max_{z in [0,d]} |phi(z)| = 1.
    The repeated antiderivatives are those returned by symbolic integration, so that
    phi(z) is proportional to |z - 1/2|^{2s+3}.
    """
    p = 2 * s + 3
    return lambda T: 1.0 + (np.abs(T.sum(axis=1) / d - 0.5) / (d - 0.5)) ** p


def exact_integral(d: int, s: int) -> float:
    """Exact value of the integral of g over [0,1]^d, via the Irwin-Hall density of sum(t)."""
    p = 2 * s + 3
    S = sp.symbols("S", real=True)
    def irwin_hall(k):  # density of the sum of d uniforms on [k, k+1]
        return sum((-1) ** j * sp.binomial(d, j) * (S - j) ** (d - 1) for j in range(k + 1)) / sp.factorial(d - 1)
    total = sp.Integer(0)
    half = sp.Rational(d, 2)
    for k in range(d):
        for lo, hi in [(k, min(k + 1, half)), (max(k, half), k + 1)]:
            lo, hi = sp.nsimplify(lo), sp.nsimplify(hi)
            if hi <= lo:
                continue
            sign = -1 if hi <= half else 1
            total += sp.integrate((sign * (S / d - sp.Rational(1, 2))) ** p * irwin_hall(k), (S, lo, hi))
    value = 1 + total / sp.Rational(2 * d - 1, 2) ** p
    return float(sp.N(value, 30))


def riemann_sum(X: np.ndarray, g, offset: int = 0) -> np.ndarray:
    """Riemann midpoint rule with ceil(1 / x_i) - offset bins in coordinate i."""
    out = np.zeros(X.shape[0])
    for j, x in enumerate(X):
        num_bins = np.ceil(1.0 / x).astype(int) - offset
        centres = [(np.arange(m) + 0.5) / m for m in num_bins]
        T = np.array(list(itertools.product(*centres)))
        out[j] = np.prod(1.0 / num_bins) * np.sum(g(T))
    return out


##############################################################################
# Extrapolation methods (fixed index sets, as in the paper)
##############################################################################

def index_set_spre(d: int, s: int) -> np.ndarray:
    """A = {2 alpha : |alpha| <= s}, from the expansion of the Riemann midpoint rule."""
    A = [a for a in itertools.product(range(s + 1), repeat=d) if sum(a) <= s]
    return 2 * np.array(sorted(A, key=lambda a: (sum(a), tuple(-x for x in a))), dtype=int)


def leading_terms_gre(d: int) -> np.ndarray:
    """Lead(A): the leading order terms, used in the numerical analysis-informed covariance of GRE."""
    return np.array([a for a in itertools.product(range(3), repeat=d) if sum(a) == 2], dtype=int)


# As in the original experiments, kernel parameters are optimised using 10 iterations of a trust region method.
MAX_ITERATIONS = 10


def fit_spre(A, X, Y, kernel="white", gre_base=None):
    spre = SPRE(kernel, X.shape[1], None if gre_base is None else jnp.array(gre_base))
    spre.set_normalised_data(jnp.array(X), jnp.array(Y))
    A = jnp.array(A)
    fit = spre.perform_extrapolation_optimization(A, max_iterations=MAX_ITERATIONS)
    out = spre.perform_extrapolation(fit["x"], A, return_mu_and_var=True)
    return float(np.ravel(out["mu"])[0]), float(np.ravel(out["var"])[0])


def fit_gre_numpy(X, Y, B):
    """
    GRE with a Matern-3/2 base kernel, implemented directly in NumPy/SciPy. This gives the same
    model and LOOCV criterion as sparse_pre.SPRE with gre_base = B, but avoids the very long
    JAX compilation time of the LOOCV criterion when the number of data is large (n = 64).
    """
    from scipy.optimize import minimize
    ep = 1e-16
    softplus = lambda z: np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)
    nX = ep + np.ptp(X, axis=0)
    nY = ep + np.ptp(Y)
    Xn, Yn = X / nX, Y / nY
    b = lambda Z: sum(np.prod(Z ** a, axis=1) for a in B)

    def kernel(Z1, Z2, x):
        r = np.sqrt(3) * np.sqrt(((Z1[:, None, :] - Z2[None, :, :]) ** 2).sum(-1)) / softplus(x[2])
        base = (ep + softplus(x[1])) * (1 + r) * np.exp(-r)
        return (ep + softplus(x[0])) * b(Z1)[:, None] * base * b(Z2)[None, :]

    def predict(Ztr, ytr, Zs, x):
        K = kernel(Ztr, Ztr, x)
        Kinv = np.linalg.inv(K)
        ks = kernel(Ztr, Zs, x)
        V = np.ones((len(ytr), 1))
        vs = np.ones((1, len(Zs)))
        r = vs - V.T @ Kinv @ ks
        M = np.linalg.inv(V.T @ Kinv @ V)
        beta = M @ (V.T @ Kinv @ ytr)
        mu = ks.T @ Kinv @ ytr + r.T @ beta
        var = kernel(Zs, Zs, x) - ks.T @ Kinv @ ks + r.T @ M @ r
        return mu[0], max(var[0, 0], 1e-300)

    def loocv(x):
        total = 0.0
        for i in range(len(Yn)):
            keep = np.arange(len(Yn)) != i
            mu, var = predict(Xn[keep], Yn[keep], Xn[[i]], x)
            total += 0.5 * np.log(2 * np.pi * var) + 0.5 * (Yn[i] - mu) ** 2 / var
        return total

    x_opt = minimize(loocv, np.ones(3), method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-9}).x
    mu, var = predict(Xn, Yn, np.zeros((1, X.shape[1])), x_opt)
    return float(nY * mu), float(nY ** 2 * var)


def fit_mre(A, X, Y):
    return float(np.ravel(MRE.MRE(jnp.array(A), jnp.array(X), jnp.array(Y))["mu"])[0])


##############################################################################
# Experiments
##############################################################################

def run_illustration(seed: int = 1) -> None:
    """Figure 1: MRE, GRE and SPRE applied to the d = 2, s = 1 task."""
    d, s = 2, 1
    g = integrand(d, s)
    f = lambda X: 1000 * (riemann_sum(X, g, offset=1) - 1)   # rescaled for visualisation
    f0 = 1000 * (exact_integral(d, s) - 1)
    A = index_set_spre(d, s)
    B = leading_terms_gre(d)
    from math import factorial
    n = {"MRE": len(A),                                 # minimum for MRE
         "GRE": ((2 ** d) * factorial(d)) ** d,         # (2^d d!)^d, as required in Theorem 2
         "SPRE": len(A) + 1}                            # one more than the minimum, for LOOCV
    rng = np.random.default_rng(seed)
    out = {"f0": f0}
    for method in ["MRE", "GRE", "SPRE"]:
        X = (1 / 3) * (0.5 + 0.5 * rng.uniform(size=(n[method], d)))
        Y = f(X)
        if method == "MRE":
            mu, var = fit_mre(A, X, Y), None
        elif method == "GRE":
            mu, var = fit_gre_numpy(X, Y, B)
        else:
            mu, var = fit_spre(A, X, Y, "white")
        out[method] = {"X": X.tolist(), "Y": Y.tolist(), "mu": mu, "var": var}
        print(f"[illustration] {method}: n = {len(Y)}, estimate = {mu}, truth = {f0}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "illustration.json").write_text(json.dumps(out, indent=1))


def run_convergence() -> None:
    """Figure 2: convergence of MRE, GRE and SPRE as h -> 0, for d in {1,2,3} and s in {0,1}."""
    X0 = {1: 0.5 / np.array([[1], [2], [3], [4]], dtype=float),
          2: 0.5 / np.array([[1, 1], [1, 2], [2, 1], [2, 2], [1, 3], [3, 1]], dtype=float),
          3: 0.5 / np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2],
                             [2, 2, 1], [2, 1, 2], [1, 2, 2], [2, 2, 2]], dtype=float)}
    rows = []
    for d in [1, 2, 3]:
        h_min = (2.0 ** -17) ** (1 / d)
        h_vals = [h for h in 2.0 ** np.arange(-19, 1) if h >= h_min]
        for s in [0, 1]:
            g = integrand(d, s)
            f0 = exact_integral(d, s)
            A = index_set_spre(d, s)
            B = leading_terms_gre(d)
            for h in h_vals:
                X = h * X0[d]
                Y = riemann_sum(X, g)
                nn = int(np.argmin(np.linalg.norm(X, axis=1)))
                mu_mre = fit_mre(A, X, Y)
                mu_gre, var_gre = fit_spre(np.zeros((1, d), dtype=int), X, Y, "Matern3/2", gre_base=B)
                mu_spre, var_spre = fit_spre(A, X, Y, "white")
                rows.append({
                    "d": d, "s": s, "h": h, "true_value": f0,
                    "abs_err_baseline": max(EPS, abs(Y[nn] - f0)),
                    "abs_err_MRE": max(EPS, abs(mu_mre - f0)),
                    "abs_err_GRE": max(EPS, abs(mu_gre - f0)),
                    "abs_err_SPRE": max(EPS, abs(mu_spre - f0)),
                    "rel_err_GRE": (mu_gre - f0) / np.sqrt(var_gre),
                    "rel_err_SPRE": (mu_spre - f0) / np.sqrt(var_spre),
                })
                print(f"[convergence] d = {d}, s = {s}, h = {h:.3e}: SPRE {rows[-1]['abs_err_SPRE']:.2e}, "
                      f"GRE {rows[-1]['abs_err_GRE']:.2e}, MRE {rows[-1]['abs_err_MRE']:.2e}", flush=True)
                RESULTS.mkdir(parents=True, exist_ok=True)
                import pandas as pd
                pd.DataFrame(rows).to_csv(RESULTS / "convergence.dat", sep="\t", index=False)


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("illustration", "all"):
        run_illustration()
    if part in ("convergence", "all"):
        run_convergence()
