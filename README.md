# Sparse Probabilistic Richardson Extrapolation

This work is a follow up to the [Probabilistic Richardson Extrapolation](https://academic.oup.com/jrsssb/article/87/2/457/7933067) paper.

## About

Almost all numerical tasks can be viewed as a type of extrapolation, where a key accuracy or tolerance setting is adjusted. This viewpoint allows for better ways to measure uncertainty and design experiments, and can even speed up how fast numerical methods converge.

Previous research showed a method called Probabilistic Richardson Extrapolation, which uses simulations at different accuracy levels to speed up large simulations — like those modeling a full heart. However, this approach needed a huge amount of data as the number of variables increased, making it impractical for complex problems.

The new method, called Sparse Probabilistic Richardson Extrapolation, is both simpler and more powerful. It introduces the concepts of effective dimension and extrapolation sparsity, which apply to many modern numerical techniques and help significantly reduce the amount of data required.

### Project Team

| Name  | Role | Affiliation
| ------------- | ------------- | ------------- |
| Chris Oates  | PI | Newcastle University  |
| Richard Howey | RSE  | Newcastle Universtiy  |

## Built With

[Python 3](https://www.python.com)

## Getting Started

The below documentation refers to the Python version of the SPRE code. There is also some MatLab code available of the SPRE method.

### Prerequisites

The Python versions used during development were **Python 3.12.6** and **3.13.9**. You can find the different versions of Python [here](https://www.python.org/downloads/).

### Installation

Clone or download the code using the green button on the top right of the home page of the GitHub [page](https://github.com/NewcastleRSE/sparse-probabilistic-richardson-extrapolation). 

### Running Locally

1. Set up a Python virtual environment in the root of the cloned repo. If you have multiple Python versions on your computer, you may need to specify the Python version (e.g., `python3.12 -m venv .venv`).
```
python -m venv .venv
```
2. Activate the virtual environment.
```
source .venv/bin/activate
```
3. Install required Python modules
```
pip install -r requirements.txt
```

4.  Copy the file `extrapaths.pth` file to `.venv/lib/pythonX.Y/site-packages/extrapaths.pth`, which `venv` uses to expand its module search path so that the tests can be ran. If you are running Windows then copy the file `extrapaths_win.pth` file to `.venv/lib/site-packages/extrapaths_win.pth`

### Running SPRE

The file `example.py` contains a simple example of running the SPRE method. From the root directory run:

```
python ./src/example.py
```

The `extrapolation` function is used to estimate f(0) from input-output training data (X, Y). The default setting is to use SPRE with a white kernel. The methods GRE and MRE can also be used, and the possible kernels are "Gaussian", "GaussianARD", "Matern1/2", "Matern3/2" and "white".

### Code Documentation

There is some code documentation automatically created from the docstrings (comments for methods etc.) that is available in `site\index.html` or on the GitHub page [here](https://github.com/NewcastleRSE/sparse-probabilistic-richardson-extrapolation/blob/main/site/index.html) which may be useful.

## Models

A few models are included in this repository to demostrate the application of the SPRE method.

### Model Code

The models are written in object-oriented Python code with `src/models/base_model.py` providing a parent class providing all the methods necessary with simulating models and organising data for use with SPRE. Users of this code may find this useful to write their own model of interest as a subclass similar to how the models present here have been. In particular, each analysis of a method against a model with certain setting is given in a `json` parameter file for ease of reproducibility.

### Running a Model
A single model simulation can be ran with the `simulate_model.py` script, a parameter file and the number of the simulation, for example:

```
python .\src\simulate_model.py .\data\mujoco\input_two_spheres_mp4.json 1
```

This will run the first Two Spheres model for the first simulation and will produce some screen shots and a video. For this parameter file there is only one set of parameters in X and only one value of h so that there is only one possible model that can run. When there are multiple parameter sets in X and multiple values for h there are more possibilities. For example, if X had 8 parameter sets and h has 10 values there are 80 possible models that could run.

This option can be useful for running many model simulations in parallel on an HPC machine when using the cache option. These cached values can then be used later to speed up any analyses.

### Running a SPRE Analysis
A SPRE analysis can be ran with the `run_model_analysis.py` script and a parameter file, for example:

```
python ./src/run_model_analysis.py ./data/cubic/input_cubic_1.json
```

### Model Parameter files

When a parameter file is given, each parameter sets an object variable with the given value. If a parameter is not given the default value will be used which is set in the class constructor.

The following is a description of the parameters:

| Parameter | Description |
|-----------|-------------|
| model_name | Name of the simulation model being used. This should be in lowercase where the model class should be called "<model_name>Model" |
| description | Description of the simulation scenario. |
| total_time | Total simulation time (in seconds). |
| X | Array of sample parameter vectors used as input points for the simulation or evaluation. |
| h_values | Sequence of values, typically between 0 and 1, used to multiple values of X with to run a series of SPRE estimates (or GRE/MRE). |
| final_tols | List of model parameters used to determine convergence/accuracy. |
| evaluation | True/False flag indicating whether evaluation of results for accuracy should be performed. |
| use_offset_model | Enables the use of an offset model for improved extrapolation evaluation. |
| use_model_cache | Enables caching of model outcome values to improve performance. Saves in cache results if a new simulation, otherwise uses previous result. |
| max_order | Maximum order to use for the basis, A, when doing SPRE optimisation. |
| results_filename | Output file where the main simulation results are stored. |
| results_eval_filename | File storing evaluation results such as absolute errors. |
| results_fx_filename | File storing evaluated function values. X and y values used at each f(0) estimation step. |
| results_eval_plot_filename | Filename for the generated plot of evaluation errors. |
| results_plot_filename | Filename for the generated plot of estimated values with errorbars of 1 standard deviation. |
| results_bases_filename | File storing basis data used during extrapolation or evaluation. |
| final_mp4_filename | Optional filename for a rendered simulation video (MP4). |
| extrapolation_name | Name of the extrapolation method used (e.g., SPRE). |
| extrapolation_kernel | Kernel type used within the extrapolation method. |

If any of the results files are not set or set to an empty string, then that results file or results plot is not saved.

### 3D Models

Additional parameters specific to the 3D MuJoCo physics models:

| Parameter | Description |
|-----------|-------------|
| model_file | XML file containing the MuJoCo model definition. |
| use_dt | Flag indicating whether a fixed timestep `dt` should be used. |
| use_solver_reference | Enables the use of a custom solver reference tolerance. |
| use_solver_impedance | Enables the use of a custom solver impedance parameter. |
| dt | Default value if not varying. Simulation timestep size used for integration. |
| solver_reference | Default value if not varying. Reference tolerance parameter for the solver. |
| solver_impedance | Default value if not varying. Impedance parameter used by the solver for constraint handling. |

Further parameters can be founf in the constructor for this class.

### Flock Model

Additional parameters specific to the Flock model:

| Parameter | Description |
|-----------|-------------|
| seed | Random seed used to ensure reproducible simulations. |
| n_agents | Number of agents participating in the flocking simulation. |
| use_dt | Flag indicating whether a fixed timestep `dt` should be used. |
| use_repulsion_softening | Enables the use of a softening parameter to stabilize repulsive interactions between agents. |
| use_cutoff_width | Enables the use of a cutoff width, smoothing the attraction of agents. |
| dt | Default value if not varying. Simulation timestep size used for steps. |
| cutoff_width | Default value if not varying. Smoothing the attraction of agents near boundary for attraction. |
| repulsion_softening | Default value if not varying. Softening factor applied to repulsive forces to avoid singularities or extreme forces at short distances. |

Further parameters can be founf in the constructor for this class.

### Adding Your Own Model

To add your own model the easiest way to do this is to use the Cubic model, `src/models/models_cubic.py`, as a template and update it for your model.

1. Copy `src/models/models_cubic.py` to `src/models/models_your_idea.py`
1. Name the class of the model `YourIdeaModel(Model)`. That is, in camel case with `Model` afterwards.
1. Update the class to simulate your model and update all of the methods.
1. In the `data` directory create a directory called `your_idea`. All results and plots will be added to `data\your_idea\results`, and if you use the cache, model outcomes will be saved to `data\your_idea\cache`.
1. Add parameter `json` files in the `data\your_idea` directory.
1. Update `src/models_utils.py` to add the line `from models.models_your_idea import *` to import your model.

After following these steps you should be able to run your analyses as above.

### Producing Plots

All of the plots produce for the empirical evaluation of the SPRE method in the paper can be produced by running:

```
python .\src\plot_analysis_results.py
```

## Files

Below is the directory structure of repository. Only a selection of directories and files are shown.

```
.
├── data                                   # Models that were used to evaulate SPRE
│   ├── cubic                              # A trivial cubic equation model      
│   │   ├── input_cubic_1.json             # Parameter file used for cubic model analysis
│   │   └── results                        # Directory of results for cubic model
│   ├── flock                              # Directory of Flock model
│   │   ├── input_flock_321.json           # Parmeter file to do SPRE analysis with white kernel of Flock model
│   │   ├── input_flock_mp4.json           # Parameter file to produce a mp4 video
│   │   └── results
│   │       └── flock1.mp4                 # Video of the Flock model
│   ├── mujoco                             # 3D physics model using MuJoCo Python library
│   │   ├── cache                          # Cache of final outcomes (y values) of models
│   │   ├── input_many_shapes_203.json     # Parameter file to do SPRE analysis with white kernel of Five Shapes model  
│   │   ├── input_many_shapes_mp4.json     # Video of the Five Shapes model
│   │   ├── input_two_spheres_300.json     # Parameter file to do SPRE analysis with white kernel of Two Spheres model  
│   │   ├── input_two_spheres_mp4.json     # Video of the Two Spheres model
│   │   ├── many_shapes.xml                # MuJoCo 3D world setup file for the Five Shapes model
│   │   ├── results
│   │   │   ├── loocv_plots                # Leave-one-out cross validation plots, if output during SPRE analysis
│   │   │   ├── many_shapes.mp4            # Videos of the 3D models
│   │   │   └── two_spheres.mp4
│   │   └── two_spheres.xml                # MuJoCo 3D world setup file for the Two Spheres model
│   └── plots                              # Directory of plots used for demostration/publication of the SPRE method 
├── docs                                   # Files used to create automatically created code documentation
├── extrapaths.pth                         # Files needed to set paths to allow unit tests to find the correct files
├── extrapaths_win.pth
├── matlab_code                            # Original MatLab code of the SPRE method
│   ├── experiment_1.m
│   ├── experiment_2.m
│   ├── experiment_3.m
│   ├── export_fig
│   ├── GRE_stepwise.m
│   ├── helper_functions
│   │   ├── AutoDiff                       # Directory of automatic differentiation code
│   │   ├── cellsum.m
│   │   ├── cwaitbar
│   │   │   ├── cwbar.m
│   │   │   └── license.txt
│   │   ├── remove_row.m
│   │   ├── shade_background.m
│   │   ├── softplus.m
│   │   ├── stepwise.m
│   │   ├── tight_subplot.m
│   │   └── white.m
│   ├── kernel.m
│   ├── MRE.m
│   ├── SPRE.m                            # SPRE function in MatLab code
│   ├── SPRE_opt.m
│   ├── SPRE_stepwise.m
│   └── unisolvent.m
├── README.md                             # This file
├── requirements.txt                      # Python library requirements
├── site                                  # Code documentation create automatically from docstrings 
├── src                                   # Source code of Python code
│   ├── example.py                        # Example of applying the SPRE method
│   ├── models                            # Models used for SPRE analysis
│   │   ├── base_model.py                 # Parent class for models
│   │   ├── models_cubic.py               # Simple cubic model
│   │   ├── models_flock.py               # 2D Flocking model
│   │   ├── models_mujoco.py              # 3D physics model using MuJoCo library
│   │   └── models_utils.py               # Helper functions
│   ├── plot_analysis_results.py          # Python script which when ran produces all plots used in SPRe paper
│   ├── run_model_analysis.py             # Run a SPRE analysis given a parameter file for models written with above model code
│   ├── simulate_model.py                 # Run one model given a parameter file and the number of model to run (e.g. 10th)
│   └── sparse_pre                        # The Python code to do the actual SPRE analysis
│       ├── extrapolation.py              # Wrapper function to call the SPRE method for given input-output data and options
│       ├── helper_functions.py           # Various helper functions
│       ├── MRE.py                        # The MRE method
│       └── SPRE.py                       # Python code for the SPRE method
└── test
    └── test_sparse_pre.py                # Unit test code for the SPRE code
```

### Running Tests

From the `sparse-probabilistic-richardson-extrapolation` directory all unit tests can be ran with:

`python -m unittest -v test.test_sparse_pre`

This tests the Python code against the original MatLab code.

## Acknowledgements
This work was funded by a grant from the UK Research Councils, EPSRC grant ref. EP/W019590/1, “Harnessing the Power of Stein Discrepancies in Bayesian Computation”.
