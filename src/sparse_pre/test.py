# Python modules
import jax.numpy as jnp
from jax import random

# Application modules
from extrapolation import extrapolation

# Set random seed
key = random.key(2)

# Parameters
d = 2  # data dimension
n_train = 5  # number of training data

# Generate training data
X = random.uniform(key, shape=(n_train, d), minval = 0.0, maxval = 1.0)  # training inputs
Y = 3 + jnp.sin(X[:, 0]) + jnp.array(X[:, 1]) #jnp.sin(X[:, 1])  # training outputs

# Define options
options = {
    "name": "SPRE",
    "k_name":  "Gaussian" #"GaussianARD" #"Matern3/2" #"Matern1/2" #"Gaussian"
}

# Assume extrapolation is a defined function returning a dict with 'mu' and 'var'
out = extrapolation(X, Y, options)

print(f"predict f(0) = {out['mu']} +/- {jnp.sqrt(out['var'])}")

# Display data
print("X =")
print(X)
print("Y =")
print(Y)