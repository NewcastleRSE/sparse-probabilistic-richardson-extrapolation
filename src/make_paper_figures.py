##############################################################################
# Produce every data figure in the paper "Sparse Probabilistic Richardson
# Extrapolation" in a common style, as PDF files in data/plots/paper/.
#
# Usage, from the root directory:
#   python ./src/make_paper_figures.py
#
# The case-study figures are drawn from the stored results in data/*/results.
# The figures for the cubature and toy experimental design examples are drawn
# from data/cubature/results and data/toy_design/results, which are produced
# by src/run_cubature_experiments.py and src/run_toy_design.py respectively.
##############################################################################

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

from paper_figures.style import use_paper_style
from paper_figures import case_studies as cs

OUT = Path("data/plots/paper")


def main(which=("case_studies", "cubature", "toy_design")) -> None:
    use_paper_style()
    OUT.mkdir(parents=True, exist_ok=True)
    if "case_studies" in which:
        for name in cs.CASE_STUDIES:
            cs.plot_abs_errors(name, OUT / f"{name}_abs_errors.pdf")
            cs.plot_error_bars(name, OUT / f"{name}_error_bars.pdf")
            for method in cs.SPRE_KERNELS:
                cs.plot_index_sets(name, method, OUT / f"{name}_index_sets_{method.split('_')[1]}.pdf")
            print(f"Done: {name}")
        cs.plot_sequential_design(OUT / "two_spheres_sequential_design.pdf")
    if "cubature" in which:
        from paper_figures import cubature
        cubature.plot_illustration(OUT / "cubature_illustration.pdf")
        cubature.plot_convergence(OUT / "cubature_convergence.pdf")
    if "toy_design" in which:
        from paper_figures import toy_design
        toy_design.plot_toy_design(OUT / "toy_design.pdf")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) if len(sys.argv) > 1 else ("case_studies", "cubature", "toy_design"))
