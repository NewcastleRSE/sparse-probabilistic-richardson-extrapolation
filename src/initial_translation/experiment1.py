import numpy as np

# Set random seed
np.random.seed(0)

# Parameters
d = 2  # data dimension
n_train = 5  # number of training data

# Generate training data
X = np.random.rand(n_train, d)  # training inputs
Y = 3 + np.sin(X[:, 0]) + np.sin(X[:, 1])  # training outputs

# Define options
options = {
    "name": "SPRE",
    "k_name": "Matern1/2"
}

# Assume extrapolation is a defined function returning a dict with 'mu' and 'var'
out = extrapolation(X, Y, options)

print(f"predict f(0) = {out['mu']} +/- {np.sqrt(out['var'])}")

# Display data
print("X =")
print(X)
print("Y =")
print(Y)