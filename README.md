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

The current Python version is **Python 3.12**, and the same version should be used for local testing and development. You can find the correct version of Python [here](https://www.python.org/downloads/).

### Installation

1. Set up Python virtual environment `python3 -m venv .venv`. If you have multiple Python versions on your computer, you may need to specify the Python version (e.g., `python3.12 -m venv .venv`).
2. Run venv with `source .venv/bin/activate`
3. Install dependencies `python -m pip install -r requirements-dev.txt`

### Running Locally

1. Set up a Python virtual environment in the root of the cloned repo.
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

### Running Tests

From `sparse-probabilistic-richardson-extrapolation/src` directory unit tests can be ran with (for example):

`python -m unittest -v test.test_sparse_pre`

## Deployment

TBA

## Contributing

### Main Branch
Protected and can only be pushed to via pull requests. It should be considered stable and a representation of production code.

### Dev Branch
Should be considered fragile; code should compile and run, but features may be prone to errors.

## Acknowledgements
This work was funded by a grant from the UK Research Councils, EPSRC grant ref. EP/W019590/1, “Harnessing the Power of Stein Discrepancies in Bayesian Computation”.
