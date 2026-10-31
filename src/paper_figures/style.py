##############################################################################
# Shared plotting style for all figures in the paper
# "Sparse Probabilistic Richardson Extrapolation".
#
# Every figure is drawn at its final printed size, so that fonts match the
# text of the paper (12pt article, 0.8in margins on A4 => text width 6.67in).
##############################################################################

import matplotlib as mpl
import matplotlib.pyplot as plt

TEXT_WIDTH = 6.67          # inches
FONT_SIZE = 9              # pt, for tick labels, legends and annotations

def use_paper_style() -> None:
    mpl.rcdefaults()
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE + 1,
        "axes.titlesize": FONT_SIZE + 1,
        "legend.fontsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#52514e",
        "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "axes.grid": True,
        "grid.color": "#d9d8d4",
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })

# Fixed categorical colours (validated for colour-vision deficiency on a white
# background); every method keeps the same colour and marker in every figure.
NEUTRAL = "#8c8b86"
METHODS = {
    "SPRE_white":    dict(color="#2a78d6", marker="o", label=r"SPRE, white noise"),
    "SPRE_Matern12": dict(color="#eb6834", marker="s", label=r"SPRE, Mat\'{e}rn-$\frac{1}{2}$"),
    "SPRE_Matern32": dict(color="#1baf7a", marker="^", label=r"SPRE, Mat\'{e}rn-$\frac{3}{2}$"),
    "SPRE_Gaussian": dict(color="#eda100", marker="D", label=r"SPRE, Gaussian"),
    "MRE":           dict(color="#e87ba4", marker="v", label=r"MRE"),
    "GRE":           dict(color="#008300", marker="P", label=r"GRE"),
    "raw":           dict(color=NEUTRAL,   marker=None, label=r"raw estimates $f(h\mathbf{x}_i)$"),
}

def style_log_axes(ax) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major")
    ax.grid(False, which="minor")
