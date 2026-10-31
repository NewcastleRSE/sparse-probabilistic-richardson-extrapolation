# MATLAB code

This directory contains the original MATLAB implementation of SPRE (`SPRE.m`, `SPRE_opt.m`, `SPRE_stepwise.m`), GRE (`GRE_stepwise.m`) and MRE (`MRE.m`), on which the Python implementation in `src/sparse_pre/` is based. The Python unit tests in `test/` check the Python implementation against outputs of this code.

The Python implementation is the reference implementation for the paper. The scripts `experiment_1.m`, `experiment_2.m` and `experiment_3.m` are early versions of the cubature and toy experimental design experiments, and are superseded by `src/run_cubature_experiments.py` and `src/run_toy_design.py`; in particular, the computational budget in `experiment_3.m` constrains the cost of each new simulation, whereas in the paper (and in `run_toy_design.py`) it constrains the total cost of each new batch of simulations.

## Third-party code

The following third-party code is included for convenience and is distributed under its own licence terms:

| Directory / file | Author | Licence |
| --- | --- | --- |
| `export_fig/` | Oliver Woodford and Yair Altman | BSD (see `export_fig/LICENSE`) |
| `helper_functions/AutoDiff/` | Martin de La Gorce | FreeBSD (see the header of `AutoDiff.m`) |
| `helper_functions/cwaitbar/` | Amal Joy | BSD (see `helper_functions/cwaitbar/license.txt`) |
| `helper_functions/tight_subplot.m` | Pekka Kumpulainen | MATLAB Central File Exchange licence (BSD) |
