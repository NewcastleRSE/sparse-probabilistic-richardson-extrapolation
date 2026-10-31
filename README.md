# Sparse Probabilistic Richardson Extrapolation

Python implementation of **Sparse Probabilistic Richardson Extrapolation (SPRE)**, together with the code and stored results needed to reproduce every experiment in the paper

> C. J. Oates, R. Howey and T. Karvonen. *Sparse Probabilistic Richardson Extrapolation.* (Manuscript under review.)

## About

Almost every numerical task can be cast as extrapolation with respect to the fidelity or tolerance parameters of a consistent numerical method. This perspective enables probabilistic uncertainty quantification and experimental design, and can be used to accelerate the convergence of numerical methods.

Earlier work on *Probabilistic Richardson Extrapolation* (Gauss–Richardson Extrapolation, GRE) required an amount of data that grows rapidly with the number of discretisation parameters. SPRE instead exploits *extrapolation sparsity*: the index set of the error expansion is learned from data, so that far fewer simulations are needed. The repository implements SPRE, GRE and Multivariate Richardson Extrapolation (MRE), with the white noise, Matérn-1/2, Matérn-3/2, Gaussian and Gaussian ARD kernels.

### Project team

| Name          | Role | Affiliation                                      |
| ------------- | ---- | ------------------------------------------------ |
| Chris Oates   | PI   | Newcastle University                             |
| Richard Howey | RSE  | Newcastle University                             |
| Toni Karvonen | Co-I | Lappeenranta–Lahti University of Technology LUT  |

## Installation

The code was developed with **Python 3.12** and **3.13**.

```
git clone https://github.com/NewcastleRSE/sparse-probabilistic-richardson-extrapolation.git
cd sparse-probabilistic-richardson-extrapolation
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The modules live in `src/`, which must be on the Python path. Either set it for the current shell,

```
export PYTHONPATH=src              # on Windows: set PYTHONPATH=src
```

or make it permanent for the virtual environment by copying `extrapaths.pth` to `.venv/lib/pythonX.Y/site-packages/` (on Windows, copy `extrapaths_win.pth` to `.venv\Lib\site-packages\`).

All commands below are run from the root directory of the repository.

`src/make_paper_figures.py` typesets its labels with LaTeX, so it additionally needs a LaTeX installation that includes the `cm-super`, `amsmath` and `bm` packages (e.g. TeX Live or MiKTeX).

## Quick start

```
python ./src/example.py
```

`example.py` estimates f(0) from a small input–output data set (X, y) using the `extrapolation` function in `src/sparse_pre/extrapolation.py`:

```python
from sparse_pre.extrapolation import extrapolation
out = extrapolation(X, y, {"name": "SPRE", "k_name": "white"})
print(out["mu"], out["var"])        # predictive mean and variance for f(0)
```

The method (`"name"`) is one of `"SPRE"` (default), `"GRE"` or `"MRE"`, and the kernel (`"k_name"`) is one of `"white"` (default), `"Matern1/2"`, `"Matern3/2"`, `"Gaussian"` or `"GaussianARD"`. For SPRE and GRE the index set and the kernel parameters are selected by stepwise leave-one-out cross-validation, as described in the paper. The selected index set and kernel parameters are returned as `out["A"]` and `out["x"]`.

## Reproducing the paper

All data figures in the paper are drawn from the stored results in `data/*/results/`, in a common style, by

```
python ./src/make_paper_figures.py
```

which writes PDF files to `data/plots/paper/`. The table below lists the script that produces the stored results behind each figure.

| Figure(s) in the paper | Results produced by | Figure file(s) in `data/plots/paper/` |
| --- | --- | --- |
| Cubature illustration (MRE, GRE, SPRE) | `python ./src/run_cubature_experiments.py illustration` | `cubature_illustration.pdf` |
| Cubature convergence acceleration | `python ./src/run_cubature_experiments.py convergence` | `cubature_convergence.pdf` |
| Toy sequential experimental design | `python ./src/run_toy_design.py` | `toy_design.pdf` |
| Two Spheres 3D model | `python ./src/run_model_analysis.py ./data/mujoco/input_two_spheres_300[_<method>].json` | `two_spheres_abs_errors.pdf`, `two_spheres_error_bars.pdf`, `two_spheres_index_sets_*.pdf` |
| Sequential design for Two Spheres | `python ./src/run_sequential_design.py ./data/mujoco/input_two_spheres_design.json` | `two_spheres_sequential_design.pdf` |
| Five Shapes 3D model | `python ./src/run_model_analysis.py ./data/mujoco/input_many_shapes_203[_<method>].json` | `five_shapes_*.pdf` |
| Agent-based flocking model | `python ./src/run_model_analysis.py ./data/flock/input_flock_321[_<method>].json` | `flock_*.pdf` |

Here `<method>` is one of `Matern12`, `Matern32`, `Gaussian`, `MRE` or `GRE`; the file without a suffix is SPRE with the white noise kernel. The Five Shapes model is called `many_shapes` in the code and data files.

The three-panel simulation snapshots of the case studies are made from the frames saved by `simulate_model.py` (see below) with

```
python ./src/plot_analysis_results.py
```

which writes them to `data/plots/`. The videos referred to in the paper are `data/mujoco/results/two_spheres.mp4`, `data/mujoco/results/many_shapes.mp4` and `data/flock/results/flock1_trail.mp4`.

**Run times.** The cubature and toy design experiments can be run on a laptop (the cubature illustration takes under a minute and the toy design about ten minutes). The case studies require many long simulations at very small discretisation parameters and were run on a high-performance computing cluster; simulator outputs are cached in `data/<model>/cache/` (not included in the repository) so that repeated analyses reuse them.

**Quantity of interest in the 3D case studies.** In the parameter files for the Two Spheres and Five Shapes case studies, `"burn_in_time": 5.0` simulates the system for 5 seconds before the measured period of `"total_time": 5.0` seconds, so the quantity of interest is the distance of the red sphere from the origin at (approximately) t = 10. This reproduces the stored results exactly. The sequential design experiment (`input_two_spheres_design.json`) has no burn-in, so its quantity of interest is the same distance at t = 5.

## Models

A few models are included to **demonstrate** the application of SPRE. They are written as Python classes, with `src/models/base_model.py` providing a parent class that contains everything needed to simulate a model and organise the data for SPRE. Each analysis of a method applied to a model is defined in a `json` parameter file, for ease of reproducibility.

### Running a single simulation

```
python ./src/simulate_model.py ./data/mujoco/input_two_spheres_mp4.json 1
```

runs the first simulation defined in the parameter file; this example saves the Two Spheres video and screenshots. When the parameter file has several input vectors in `X` and several values in `h_values`, the second argument selects which simulation to run (e.g. 8 inputs and 10 values of h give 80 simulations). Together with the cache, this can be used to run many simulations in parallel on an HPC cluster.

### Running an analysis

```
python ./src/run_model_analysis.py ./data/cubic/input_cubic_1.json
```

### Parameter files

Each parameter sets an attribute of the model object; parameters that are not given take the default value set in the class constructor.

| Parameter                  | Description |
| -------------------------- | ----------- |
| model_name                 | Name of the simulation model, in lowercase; the model class is called `<ModelName>Model`. |
| description                | Description of the simulation scenario. |
| total_time                 | Length of the measured simulation period (in seconds). |
| X                          | Input vectors (discretisation parameters) of the reference design. |
| h_values                   | Scalings h; the design used for each SPRE (or GRE/MRE) estimate is h X. |
| final_tols                 | Offset x_0: the model analysed is f(x + x_0), and the ground truth is f(x_0). |
| evaluation                 | Whether to evaluate the accuracy of the estimates against the ground truth. |
| use_offset_model           | Whether to analyse the offset model f(x + x_0). |
| use_model_cache            | Whether to cache simulator outputs, and reuse cached outputs when available. |
| max_order                  | Maximum order of the index set A in stepwise selection (0 for no limit). |
| results_filename           | File for the estimates of f(0). |
| results_eval_filename      | File for the evaluation results, such as absolute errors. |
| results_fx_filename        | File for the input and output values used for each estimate of f(0). |
| results_eval_plot_filename | Plot of the absolute errors. |
| results_plot_filename      | Plot of the estimates with error bars of one standard deviation. |
| results_bases_filename     | File for the selected index sets. |
| final_mp4_filename         | Optional filename for a video of the simulation (MP4). |
| extrapolation_name         | Extrapolation method: `SPRE`, `GRE` or `MRE`. |
| extrapolation_kernel       | Kernel used by the extrapolation method. |

Results files that are not set, or set to an empty string, are not saved.

Additional parameters for the 3D MuJoCo models (`src/models/models_mujoco.py`):

| Parameter            | Description |
| -------------------- | ----------- |
| model_file           | MuJoCo XML model file, in the same directory as the parameter file. |
| burn_in_time         | Simulated time before the measured period (default 0; see above). |
| use_dt               | Whether the time step is a discretisation parameter. |
| use_solver_reference | Whether the solver reference is a discretisation parameter. |
| use_solver_impedance | Whether the solver impedance is a discretisation parameter. |
| dt                   | Time step, if not a discretisation parameter. |
| solver_reference     | Solver reference (time scale of constraint correction), if not a discretisation parameter. |
| solver_impedance     | Solver impedance (contact stiffness), if not a discretisation parameter. |

Additional parameters for the flocking model (`src/models/models_flock.py`):

| Parameter               | Description |
| ----------------------- | ----------- |
| seed                    | Random seed for the initial positions, so that the model is deterministic. |
| n_agents                | Number of agents. |
| use_dt                  | Whether the time step is a discretisation parameter. |
| use_repulsion_softening | Whether the repulsion softening is a discretisation parameter. |
| use_cutoff_width        | Whether the cutoff width is a discretisation parameter. |
| dt                      | Time step, if not a discretisation parameter. |
| repulsion_softening     | Softening of the repulsive force at short distances, if not a discretisation parameter. |
| cutoff_width            | Width of the smooth cutoff of the interaction radius, if not a discretisation parameter. |

Further parameters can be found in the constructor of each class.

### Adding your own model

The simplest template is the cubic model, `src/models/models_cubic.py`.

1. Copy `src/models/models_cubic.py` to `src/models/models_your_idea.py`.
2. Name the class `YourIdeaModel(Model)`, in camel case with `Model` appended.
3. Update the methods of the class to simulate your model.
4. Create a directory `data/your_idea` containing your `json` parameter files. Results and plots are written to `data/your_idea/results`, and cached simulator outputs to `data/your_idea/cache`.
5. Add the line `from models.models_your_idea import *` to `src/models/models_utils.py`.

### Sequential experimental design

`src/run_sequential_design.py` compares SPRE based on a fixed design with SPRE based on a sequential, cost-constrained design, as a function of the total computational cost:

```
python ./src/run_sequential_design.py ./data/mujoco/input_two_spheres_design.json [fixed|sequential|plot]
```

The settings are in the `design` entry of the parameter file and are documented at the top of the script. Because the ground truth is computed using the offset model, the actual run time of each simulation is approximately constant; a nominal cost c(x) = 1 / x_1, proportional to the number of time steps that would be required without the offset, is therefore used.

## Repository structure

```
.
├── data
│   ├── cubature/results          # Stored results of the cubature experiments
│   ├── cubic                     # Simple cubic model (example for new models)
│   ├── flock                     # Agent-based flocking model: parameter files and results
│   ├── mujoco                    # Two Spheres and Five Shapes (many_shapes) 3D models:
│   │                             #   MuJoCo XML files, parameter files and results
│   ├── toy_design/results        # Stored results of the toy experimental design
│   └── plots
│       └── paper                 # Data figures in the paper (make_paper_figures.py)
├── docs                          # Source of the code documentation (mkdocs)
├── matlab_code                   # Original MATLAB implementation (see matlab_code/README.md)
├── src
│   ├── example.py                # Minimal example of SPRE
│   ├── make_paper_figures.py     # All data figures in the paper
│   ├── paper_figures             # Plotting code used by make_paper_figures.py
│   ├── plot_analysis_results.py  # Simulation snapshot figures
│   ├── run_cubature_experiments.py
│   ├── run_model_analysis.py     # SPRE/GRE/MRE analysis defined by a parameter file
│   ├── run_sequential_design.py  # Fixed vs sequential experimental design
│   ├── run_toy_design.py         # Toy illustration of sequential experimental design
│   ├── simulate_model.py         # Run a single simulation from a parameter file
│   ├── simulate_models_timings_grid.py  # Run-time measurements for the Two Spheres model
│   ├── models                    # Simulation models (base class, cubic, flock, MuJoCo)
│   └── sparse_pre                # SPRE, GRE and MRE
│       ├── extrapolation.py      # Main entry point
│       ├── SPRE.py               # SPRE and GRE
│       ├── MRE.py                # MRE
│       └── helper_functions.py
└── test
    └── test_sparse_pre.py        # Unit tests against the original MATLAB implementation
```

## Tests

```
python -m unittest -v test.test_sparse_pre
```

The tests compare the Python implementation against outputs of the original MATLAB implementation.

## Code documentation

Documentation generated from the docstrings is available on [GitHub Pages](https://newcastlerse.github.io/sparse-probabilistic-richardson-extrapolation/). It is rebuilt automatically by GitHub Actions (`.github/workflows/docs.yml`) whenever `main` is updated. To build it locally:

```
pip install -r docs/requirements.txt
mkdocs serve        # or: mkdocs build, mkdocs gh-deploy
```

## Citation

If you use this code, please cite the paper above; citation metadata is given in `CITATION.cff`.

## Licence

The code is released under the MIT licence; see `LICENSE`. Third-party MATLAB code in `matlab_code/` is distributed under its own licence terms.

## Acknowledgements

This work was funded by EPSRC grant EP/W019590/1 and a Leverhulme Prize.
