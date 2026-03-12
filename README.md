Below is your README with:

* **All typos fixed**
* **Grammar improvements applied**
* **Unix-style paths used consistently (`./src/...`)**
* **`y` kept lowercase as requested**
* **Table markdown fixed**
* **MATLAB capitalization standardized**

I avoided changing the structure or tone beyond the corrections you asked for.

---

# Sparse Probabilistic Richardson Extrapolation

This work is a **follow-up** to the [Probabilistic Richardson Extrapolation](https://academic.oup.com/jrsssb/article/87/2/457/7933067) paper.

## About

Almost all numerical tasks can be viewed as a type of extrapolation, where a key accuracy or tolerance setting is adjusted. This viewpoint allows for better ways to measure uncertainty and design experiments, and can even speed up how fast numerical methods converge.

Previous research showed a method called Probabilistic Richardson Extrapolation, which uses simulations at different accuracy levels to speed up large simulations — like those modeling a full heart. However, this approach required a huge amount of data as the number of variables increased, making it impractical for complex problems.

The new method, called Sparse Probabilistic Richardson Extrapolation, is both simpler and more powerful. It introduces the concepts of effective dimension and extrapolation sparsity, which apply to many modern numerical techniques and help to significantly reduce the amount of data required.

### Project Team

| Name          | Role | Affiliation          |
| ------------- | ---- | -------------------- |
| Chris Oates   | PI   | Newcastle University |
| Richard Howey | RSE  | Newcastle University |

## Built With

[Python 3](https://www.python.org)

## Getting Started

The documentation below refers to the Python version of the SPRE code. There is also some MATLAB code available for the SPRE method.

### Prerequisites

The Python versions used during development were **Python 3.12.6** and **3.13.9**. You can find the different versions of Python [here](https://www.python.org/downloads/).

### Installation

Clone or download the code using the green button at the top right of the GitHub [page](https://github.com/NewcastleRSE/sparse-probabilistic-richardson-extrapolation).

### Running Locally

1. Set up a Python virtual environment in the root of the cloned repository. If you have multiple Python versions on your computer, you may need to specify the Python version (e.g., `python3.12 -m venv .venv`).

```
python -m venv .venv
```

2. Activate the virtual environment.

```
source .venv/bin/activate
```

3. Install the required Python modules.

```
pip install -r requirements.txt
```

4. Copy the file `extrapaths.pth` to `.venv/lib/pythonX.Y/site-packages/extrapaths.pth`, which `venv` uses to expand its module search path so that the tests can be run.

If you are running Windows, copy the file `extrapaths_win.pth` to `.venv/lib/site-packages/extrapaths_win.pth`.

### Running SPRE

The file `example.py` contains a simple example of running the SPRE method. From the root directory run:

```
python ./src/example.py
```

The `extrapolation` function is used to estimate f(0) from input-output training data (X, y). The default setting is to use SPRE with a white kernel. The methods GRE and MRE can also be used, and the possible kernels are `"Gaussian"`, `"GaussianARD"`, `"Matern1/2"`, `"Matern3/2"` and `"white"`.

### Code Documentation

Some code documentation is automatically created from docstrings (comments for methods, etc.) and is available in `site/index.html` or on GitHub pages [here](https://newcastlerse.github.io/sparse-probabilistic-richardson-extrapolation/).

## Models

A few models are included in this repository to **demonstrate** the application of the SPRE method.

### Model Code

The models are written in object-oriented Python code with `src/models/base_model.py` providing a parent class that includes all the methods necessary for simulating models and organising data for use with SPRE.

Users of this code may find this useful when writing their own model of interest as a subclass, similar to how the models presented here have been implemented.

In particular, each analysis of a method against a model with certain settings is defined in a `json` parameter file for ease of reproducibility.

### Running a Model

A single model simulation can be **run** with the `simulate_model.py` script, a parameter file, and the number of the simulation. For example:

```
python ./src/simulate_model.py ./data/mujoco/input_two_spheres_mp4.json 1
```

This will run the first Two Spheres model for the first simulation and will produce some screenshots and a video.

For this parameter file there is only one set of parameters in X and only one value of h, so there is only one possible model that can run. When there are multiple parameter sets in X and multiple values for h there are more possibilities.

For example, if X had 8 parameter sets and h has 10 values there are 80 possible models that could run.

This option can be useful for running many model simulations in parallel on an HPC machine when using the cache option. These cached values can then be used later to speed up analyses.

### Running a SPRE Analysis

A SPRE analysis can be **run** with the `run_model_analysis.py` script and a parameter file. For example:

```
python ./src/run_model_analysis.py ./data/cubic/input_cubic_1.json
```

### Model Parameter Files

When a parameter file is given, each parameter sets an object variable with the given value. If a parameter is not given the default value will be used, which is set in the class constructor.

The following is a description of the parameters:

| Parameter                  | Description                                                                                                                                              |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_name                 | Name of the simulation model being used. This should be in lowercase where the model class should be called "<model_name>Model".                         |
| description                | Description of the simulation scenario.                                                                                                                  |
| total_time                 | Total simulation time (in seconds).                                                                                                                      |
| X                          | Array of sample parameter vectors used as input points for the simulation or evaluation.                                                                 |
| h_values                   | Sequence of values, typically between 0 and 1, used to multiply values of X to run a series of SPRE estimates (or GRE/MRE).                              |
| final_tols                 | List of model parameters used to determine convergence or accuracy.                                                                                      |
| evaluation                 | True/False flag indicating whether evaluation of results for accuracy should be performed.                                                               |
| use_offset_model           | Enables the use of an offset model for improved extrapolation evaluation.                                                                                |
| use_model_cache            | Enables caching of model outcome values to improve performance. Saves results in the cache if a new simulation is run, otherwise uses a previous result. |
| max_order                  | Maximum order to use for the basis, A, when doing SPRE optimisation.                                                                                     |
| results_filename           | Output file where the main simulation results are stored.                                                                                                |
| results_eval_filename      | File storing evaluation results such as absolute errors.                                                                                                 |
| results_fx_filename        | File storing evaluated function values. X and y values used at each f(0) estimation step.                                                                |
| results_eval_plot_filename | Filename for the generated plot of evaluation errors.                                                                                                    |
| results_plot_filename      | Filename for the generated plot of estimated values with error bars of 1 standard deviation.                                                             |
| results_bases_filename     | File storing basis data used during extrapolation or evaluation.                                                                                         |
| final_mp4_filename         | Optional filename for a rendered simulation video (MP4).                                                                                                 |
| extrapolation_name         | Name of the extrapolation method used (e.g., SPRE).                                                                                                      |
| extrapolation_kernel       | Kernel type used within the extrapolation method.                                                                                                        |

If any of the results files are not set or are set to an empty string, then that results file or plot is not saved.

### 3D Models

Additional parameters specific to the 3D MuJoCo physics models:

| Parameter            | Description                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------------- |
| model_file           | XML file containing the MuJoCo model definition.                                              |
| use_dt               | Flag indicating whether a fixed timestep `dt` should be used.                                 |
| use_solver_reference | Enables the use of a custom solver reference tolerance.                                       |
| use_solver_impedance | Enables the use of a custom solver impedance parameter.                                       |
| dt                   | Default value if not varying. Simulation timestep size used for integration.                  |
| solver_reference     | Default value if not varying. Reference tolerance parameter for the solver.                   |
| solver_impedance     | Default value if not varying. Impedance parameter used by the solver for constraint handling. |

Further parameters can be **found** in the constructor for this class.

### Flock Model

Additional parameters specific to the Flock model:

| Parameter               | Description                                                                                                                             |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| seed                    | Random seed used to ensure reproducible simulations.                                                                                    |
| n_agents                | Number of agents participating in the flocking simulation.                                                                              |
| use_dt                  | Flag indicating whether a fixed timestep `dt` should be used.                                                                           |
| use_repulsion_softening | Enables the use of a softening parameter to stabilise repulsive interactions between agents.                                            |
| use_cutoff_width        | Enables the use of a cutoff width, smoothing the attraction of agents.                                                                  |
| dt                      | Default value if not varying. Simulation timestep size used for steps.                                                                  |
| cutoff_width            | Default value if not varying. Smoothing the attraction of agents near the boundary for attraction.                                      |
| repulsion_softening     | Default value if not varying. Softening factor applied to repulsive forces to avoid singularities or extreme forces at short distances. |

Further parameters can be **found** in the constructor for this class.

### Adding Your Own Model

To add your own model, the easiest way is to use the Cubic model, `src/models/models_cubic.py`, as a template and update it for your model.

1. Copy `src/models/models_cubic.py` to `src/models/models_your_idea.py`
2. Name the class `YourIdeaModel(Model)`, in camel case with `Model` appended.
3. Update the class to simulate your model and modify all of the methods.
4. In the `data` directory create a directory called `your_idea`. All results and plots will be added to `data/your_idea/results`, and if you use the cache, model outcomes will be saved to `data/your_idea/cache`.
5. Add parameter `json` files in the `data/your_idea` directory.
6. Update `src/models_utils.py` to add the line:

```
from models.models_your_idea import *
```

After following these steps you should be able to run your analyses as above.

### Producing Plots

All of the plots for the empirical evaluation of the SPRE method in the paper can be produced by running:

```
python ./src/plot_analysis_results.py
```

## Files

Below is the directory structure of the repository. Only a selection of directories and files are shown.

```
.
├── data                                   # Models that were used to evaluate SPRE
│   ├── cubic
│   │   ├── input_cubic_1.json
│   │   └── results
│   ├── flock
│   │   ├── input_flock_321.json           # Parameter file to do SPRE analysis with white kernel
│   │   ├── input_flock_mp4.json
│   │   └── results
│   │       └── flock1.mp4
│   ├── mujoco
│   │   ├── input_many_shapes_203.json
│   │   ├── input_many_shapes_mp4.json
│   │   ├── input_two_spheres_300.json
│   │   ├── input_two_spheres_mp4.json
│   │   ├── many_shapes.xml
│   │   ├── results
│   │   │   ├── loocv_plots
│   │   │   ├── many_shapes.mp4
│   │   │   └── two_spheres.mp4
│   │   └── two_spheres.xml
│   └── plots                              # Directory of plots used for demonstration/publication
├── docs                                   # Files used to create automatically generated documentation
├── extrapaths.pth
├── extrapaths_win.pth
├── matlab_code                            # Original MATLAB code of the SPRE method
├── README.md
├── requirements.txt
├── site                                   # Code documentation created automatically from docstrings
├── src
│   ├── example.py
│   ├── models
│   ├── plot_analysis_results.py           # Script that produces plots used in the SPRE paper
│   ├── run_model_analysis.py
│   ├── simulate_model.py
│   └── sparse_pre
└── test
    └── test_sparse_pre.py
```

### Running Tests

From the `sparse-probabilistic-richardson-extrapolation` directory all unit tests can be **run** with:

```
python -m unittest -v test.test_sparse_pre
```

This tests the Python code against the original MATLAB code.

## Acknowledgements

This work was funded by a grant from the UK Research Councils, EPSRC grant ref. **EP/W019590/1**,
“Harnessing the Power of Stein Discrepancies in Bayesian Computation”.
