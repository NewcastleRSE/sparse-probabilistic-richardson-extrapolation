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

This section is intended to list the frameworks and tools you're using to develop this software. Please link to the home page or documentation in each case.

[Framework 1](https://something.com)  
[Framework 2](https://something.com)  
[Framework 3](https://something.com)  

## Getting Started

### Prerequisites

Any tools or versions of languages needed to run code. For example, specific Python or Node versions. Minimum hardware requirements also go here.

### Installation

How to build or install the application.

### Running Locally

How to run the application on your local system. Examples of this would include `venv`, `anaconda`, `node`, `Docker` or `minikube`. 

### Running Tests

From `sparse-probabilistic-richardson-extrapolation/src` directory unit tests can be ran with:

`python -m unittest -v test.test_initial_translation`

## Deployment

Instructions on how to deploy to the staging or production systems. Examples of this would include cloud, HPC or virtual machine. Deployment should be done via GitHub Workflows but information on how these work and the different triggers should go here.

## Contributing

### Main Branch
Protected and can only be pushed to via pull requests. It should be considered stable and a representation of production code.

### Dev Branch
Should be considered fragile; code should compile and run, but features may be prone to errors.

### Feature Branches
A branch per feature that is being worked on.

https://nvie.com/posts/a-successful-git-branching-model/

## Acknowledgements
This work was funded by a grant from the UK Research Councils, EPSRC grant ref. EP/W019590/1, “Harnessing the Power of Stein Discrepancies in Bayesian Computation”.
