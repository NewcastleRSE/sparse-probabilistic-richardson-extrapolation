##############################################################################
# Figures for the cubature examples (Figures 1 and 2 of the paper), drawn from
# data/cubature/results, which is produced by src/run_cubature_experiments.py.
##############################################################################

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_figures.style import METHODS, TEXT_WIDTH, NEUTRAL, style_log_axes

RESULTS = Path("data/cubature/results")
INK = "#0b0b0b"
ESTIMATE = "#e34948"


def plot_illustration(out: Path) -> None:
    """Figure 1: MRE, GRE and SPRE applied to the two-dimensional cubature task."""
    res = json.loads((RESULTS / "illustration.json").read_text())
    f0 = res["f0"]
    allY = np.concatenate([res[m]["Y"] for m in ["MRE", "GRE", "SPRE"]] + [[f0]])
    zlo, zhi = min(allY.min(), f0) - 0.05, max(allY.max(), f0) + 0.05
    fig = plt.figure(figsize=(TEXT_WIDTH, 0.36 * TEXT_WIDTH))
    titles = {"MRE": r"MRE", "GRE": r"GRE", "SPRE": r"SPRE"}
    for k, method in enumerate(["MRE", "GRE", "SPRE"]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        X = np.array(res[method]["X"])
        Y = np.array(res[method]["Y"])
        for x, y in zip(X, Y):
            ax.plot([x[0], x[0]], [x[1], x[1]], [zlo, y], color=NEUTRAL, lw=0.5, ls=":")
        ax.scatter(X[:, 0], X[:, 1], Y, s=9, facecolor="white", edgecolor=INK, linewidth=0.6, depthshade=False,
                   label=r"data $f(\mathbf{x}_i)$")
        ax.scatter([0], [0], [f0], marker="*", s=45, color=METHODS["SPRE_white"]["color"], depthshade=False,
                   label=r"$f(\mathbf{0})$")
        mu, var = res[method]["mu"], res[method]["var"]
        ax.scatter([0], [0], [mu], marker="^", s=22, color=ESTIMATE, depthshade=False, label=r"estimate")
        if var is not None:
            sd = np.sqrt(var)
            ax.plot([0, 0], [0, 0], [mu - sd, mu + sd], color=ESTIMATE, lw=1.2, label=r"$\pm 1$ standard deviation")
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.set_zlim(zlo, zhi)
        from matplotlib.ticker import MaxNLocator
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(3))
        ax.set_xlabel(r"$x_1$", labelpad=-9)
        ax.set_ylabel(r"$x_2$", labelpad=-9)
        ax.zaxis.set_rotate_label(False)
        ax.text2D(0.97, 0.86, r"$f(\mathbf{x})$", transform=ax.transAxes, ha="right", fontsize=9)
        ax.tick_params(labelsize=6.5, pad=-3)
        ax.set_title(f"{titles[method]} ($n = {len(Y)}$)", pad=-2)
        ax.view_init(elev=22, azim=-55)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis._axinfo["grid"]["color"] = "#d9d8d4"
            axis._axinfo["grid"]["linewidth"] = 0.4
        if method == "SPRE":
            handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(l) for l in [r"$f(\mathbf{0})$", r"estimate", r"$\pm 1$ standard deviation",
                                       r"data $f(\mathbf{x}_i)$"]]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.0, right=0.98, bottom=0.12, top=0.95, wspace=0.05)
    fig.savefig(out)
    plt.close(fig)


def _shade_standard_normal(ax, ylim):
    y = np.linspace(*ylim, 400)
    dens = np.exp(-0.5 * y ** 2)
    ax.imshow(dens[:, None], extent=(*ax.get_xlim(), *ylim), origin="lower", aspect="auto",
              cmap="Greys", vmin=0, vmax=3.0, zorder=0, alpha=0.9)


def plot_convergence(out: Path) -> None:
    """Figure 2: absolute and relative errors in the d-dimensional cubature task."""
    df = pd.read_csv(RESULTS / "convergence.dat", sep="\t")
    fig, axes = plt.subplots(3, 4, figsize=(TEXT_WIDTH, 0.78 * TEXT_WIDTH))
    styles = {"baseline": dict(color=NEUTRAL, marker="s", label="baseline"),
              "MRE": METHODS["MRE"], "GRE": METHODS["GRE"],
              "SPRE": dict(METHODS["SPRE_white"], label=r"SPRE")}
    for r, d in enumerate([1, 2, 3]):
        for c, s in enumerate([0, 1]):
            g = df[(df.d == d) & (df.s == s)].sort_values("h")
            ax = axes[r, 2 * c]
            for key in ["baseline", "MRE", "GRE", "SPRE"]:
                st = styles[key]
                ax.plot(g.h, g[f"abs_err_{key}"], color=st["color"], marker=st["marker"], ms=2.8, lw=1.0,
                        markeredgecolor="white", markeredgewidth=0.3,
                        label=st["label"] if key != "SPRE" else "SPRE")
            style_log_axes(ax)
            ax.set_title(rf"$d={d}$, $s={s}$")
            if c == 0:
                ax.set_ylabel("absolute error")
            ax = axes[r, 2 * c + 1]
            ax.set_xscale("log")
            ax.set_xlim(g.h.min() / 1.6, g.h.max() * 1.6)
            vals = np.concatenate([g.rel_err_GRE.values, g.rel_err_SPRE.values])
            lim = max(3.0, 1.1 * np.nanmax(np.abs(vals)))
            _shade_standard_normal(ax, (-lim, lim))
            for key in ["GRE", "SPRE"]:
                st = styles[key]
                ax.plot(g.h, g[f"rel_err_{key}"], color=st["color"], marker=st["marker"], ms=2.8, lw=1.0,
                        markeredgecolor="white", markeredgewidth=0.3, zorder=3)
            ax.set_ylim(-lim, lim)
            ax.grid(True, which="major")
            ax.set_title(rf"$d={d}$, $s={s}$")
            if c == 0:
                ax.set_ylabel("relative error")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$h$")
    for ax in axes.ravel():
        ax.minorticks_off()
        ax.tick_params(labelsize=7.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.11, top=0.96, hspace=0.55, wspace=0.5)
    fig.savefig(out)
    plt.close(fig)
