# Python modules
import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Application modules
from sparse_pre.extrapolation import extrapolation

def simulate_grayscott_3d(
    Lx=2.0, Ly=2.0, Lz=2.0,  # domain size
    dx=0.05, dy=0.05, dz=0.05, # spatial discretization hyperparameters
    dt=1e-3, t_final=1.0,     # time discretization
    Du=0.16, Dv=0.08,         # diffusion coefficients
    F=0.035, k=0.060,         # reaction parameters
    seed=0
):
    """
    Simulate the 3D Gray-Scott reaction-diffusion system with periodic boundaries.

    Parameters
    ----------
    Lx, Ly, Lz : float
        Physical domain size in x, y, z directions.
    dx, dy, dz : float
        Spatial grid spacing (discretization hyperparameters).
    dt : float
        Time step size (discretization hyperparameter).
    t_final : float
        Final simulation time.
    Du, Dv : float
        Diffusion coefficients for u and v.
    F, k : float
        Reaction parameters.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    U, V : ndarray
        Concentrations at final time, shape (Nx, Ny, Nz).
    """
    # Grid resolution
    Nx = int(round(Lx / dx))
    Ny = int(round(Ly / dy))
    Nz = int(round(Lz / dz))

    # Mesh spacing
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz

    # Initialize fields
    np.random.seed(seed)
    U = np.ones((Nx, Ny, Nz))
    V = np.zeros((Nx, Ny, Nz))

    # Small perturbation in center region
    cx, cy, cz = Nx//2, Ny//2, Nz//2
    r = min(Nx, Ny, Nz) // 10
    U[cx-r:cx+r, cy-r:cy+r, cz-r:cz+r] = 0.50
    V[cx-r:cx+r, cy-r:cy+r, cz-r:cz+r] = 0.25
    # Add small random noise
    U += 0.05 * np.random.randn(Nx, Ny, Nz)
    V += 0.05 * np.random.randn(Nx, Ny, Nz)

    # Helper: periodic Laplacian
    def laplacian(arr, dx, dy, dz):
        return (
            (np.roll(arr, -1, axis=0) - 2*arr + np.roll(arr, 1, axis=0)) / dx**2 +
            (np.roll(arr, -1, axis=1) - 2*arr + np.roll(arr, 1, axis=1)) / dy**2 +
            (np.roll(arr, -1, axis=2) - 2*arr + np.roll(arr, 1, axis=2)) / dz**2
        )

    # Time loop
    t = 0.0
    while t < t_final - 1e-12:
        Lu = laplacian(U, dx, dy, dz)
        Lv = laplacian(V, dx, dy, dz)

        # Reaction term
        UVV = U * V * V
        U_new = U + dt * (Du * Lu - UVV + F * (1 - U))
        V_new = V + dt * (Dv * Lv + UVV - (F + k) * V)

        U, V = U_new, V_new
        t += dt

    return U, V



# Inspect central slice

if False:
    U, V = simulate_grayscott_3d(
    Lx=2.0, Ly=2.0, Lz=2.0,
    dx=0.05, dy=0.05, dz=0.05,
    dt=1e-3, t_final=5.0,
    Du=0.16, Dv=0.08,
    F=0.035, k=0.060
)
    plt.imshow(U[:, :, U.shape[2]//2], cmap="viridis")
    plt.colorbar(label="U concentration")
    plt.title("Gray-Scott U field (central slice)")
    plt.show()

# Values to try
X = np.array([[0.1, 0.1, 0.1, 0.01], [0.2, 0.2, 0.2, 0.008], [0.5, 0.5, 0.5, 0.006]])
#X = np.array([[0.05, 0.05, 0.05, 0.001]])

# Results
Y = np.array([])

# Get results, let V be y
for x in X:
    U, V = simulate_grayscott_3d(
    Lx=2.0, Ly=2.0, Lz=2.0,
    dx=x[0], dy=x[1], dz=x[2],
    dt=x[3], t_final=5.0,
    Du=0.16, Dv=0.08,
    F=0.035, k=0.060)

    # Get concentration in first cell
    y = U[0, 0, 0]
    Y = np.append(Y, y)

print(f"X = {X}")
print(f"Y = {Y}")

#exit(0)

# Apply SPRE
# Define options
options = {
    "name": "SPRE",
    "k_name":  "Gaussian", #"GaussianARD" #"Matern3/2" #"Matern1/2" #"Gaussian"
    "plot" : True
}

# Assume extrapolation is a defined function returning a dict with 'mu' and 'var'
out = extrapolation(X, Y, options)

print(f"Predict f(0) = {out['mu'][0]} +/- {np.sqrt(out['var'][0][0])}\n")

# Compute accurate answer
acc_step_x = 0.05
acc_step_t = 0.001
U, V = simulate_grayscott_3d(dx=acc_step_x, dy=acc_step_x, dz=acc_step_x, dt=acc_step_t)
y_accuarte = U[0, 0, 0]

print(f"\nAccurate prediction using dx/dy step {acc_step_x} and dt step {acc_step_t} gives f(0) = {y_accuarte}\n")

diff = abs(out['mu'][0] - y_accuarte)
print(f"This is a difference of {diff:.4f}")


