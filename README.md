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
python .\src\example.py
```

The `extrapolation` function is used to estimate f(0) from input-output training data (X, Y). The default setting is to use SPRE with a white kernel. The methods GRE and MRE can also be used, and the possible kernels are "Gaussian", "GaussianARD", "Matern1/2", "Matern3/2" and "white".

## Models

A few models are included in this repository to demostrate the application of the SPRE method.

### Model Code

The models are written in object-oriented python code with `src/models/base_model.py` providing a parent class providing all the methods necessary with simulating models and organising data for use with SPRE. Users of this code may find this useful to write their own model of interest as a subclass similar to how the models present here have been. In particular, each analysis of a method against a model with certain setting is given in a `json` parameter file for ease of reproducibility.

### Running a Model

### Running a SPRE Analysis

### Model Parameter files

The following...

### Producing Plots

All of the plots produce for the empicprical evaluation of the SPRE method in the paper can be produced by running:

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
│   ├── mujoco                             # 3D physics model using MuJoCo python library
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
│   └── sparse_pre                        # The python code to do the actual SPRE analysis
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

This tests the python code against the original MatLab code.

## Acknowledgements
This work was funded by a grant from the UK Research Councils, EPSRC grant ref. EP/W019590/1, “Harnessing the Power of Stein Discrepancies in Bayesian Computation”.
