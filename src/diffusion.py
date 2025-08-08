# Python modules
import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Application modules
from sparse_pre.extrapolation import extrapolation

def simulate_advection_diffusion_2d(
    Lx=1.0, Ly=1.0,         # domain size
    vx=1.0, vy=0.5,         # advection velocities (model parameters)
    D=0.01,                 # diffusion coefficient (model parameter)
    dx=0.02, dy=0.02,       # spatial discretization hyperparameters
    dt=0.001,               # time step hyperparameter
    t_final=0.1,            # simulation end time
    theta=0.5,              # theta-method parameter (0=explicit, 0.5=CN, 1=implicit)
    u0_func=None,           # initial condition function u0(x,y)
    show_plot=False         # Show plot or not
):
    """
    Simulate 2D advection–diffusion equation on a uniform periodic grid.

    Model parameters:
        vx, vy: advection velocities
        D: diffusion coefficient
        Lx, Ly: domain size

    Discretization hyperparameters:
        dx, dy: spatial resolution
        dt: time step
        theta: theta-method parameter
    """
    # Determine number of points
    Nx = int(round(Lx / dx))
    Ny = int(round(Ly / dy))

    # Adjust dx, dy to fit domain exactly
    dx = Lx / Nx
    dy = Ly / Ny

    # Grid
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Initial condition
    if u0_func is None:
        u = np.exp(-100 * ((X - Lx/2)**2 + (Y - Ly/2)**2))  # Gaussian blob
    else:
        u = u0_func(X, Y)

    # Helper for periodic indexing
    def periodic_roll(arr, shift, axis):
        return np.roll(arr, shift=shift, axis=axis)

    # Time integration
    t = 0.0
    while t < t_final - 1e-12:
        if theta == 0.0:
            # Fully explicit Euler
            u_x = (periodic_roll(u, -1, axis=0) - periodic_roll(u, 1, axis=0)) / (2*dx)
            u_y = (periodic_roll(u, -1, axis=1) - periodic_roll(u, 1, axis=1)) / (2*dy)
            lap = ((periodic_roll(u, -1, axis=0) - 2*u + periodic_roll(u, 1, axis=0)) / dx**2 +
                   (periodic_roll(u, -1, axis=1) - 2*u + periodic_roll(u, 1, axis=1)) / dy**2)
            Lu = -(vx*u_x + vy*u_y) + D * lap
            u = u + dt * Lu

        else:
            # θ-method: (I - θ dt L) u^{n+1} = (I + (1-θ) dt L) u^n
            ex_x = np.ones(Nx)
            ex_y = np.ones(Ny)
            # 1D second-derivative (diffusion) with periodic BC
            Dxx = diags([ex_x, -2*ex_x, ex_x], [-1, 0, 1], shape=(Nx, Nx))
            Dxx = Dxx.tolil()
            Dxx[0, -1] = 1; Dxx[-1, 0] = 1
            Dxx /= dx**2
            Dyy = diags([ex_y, -2*ex_y, ex_y], [-1, 0, 1], shape=(Ny, Ny))
            Dyy = Dyy.tolil()
            Dyy[0, -1] = 1; Dyy[-1, 0] = 1
            Dyy /= dy**2

            # 1D first-derivative (advection) with periodic BC
            Dxc = diags([-ex_x, ex_x], [-1, 1], shape=(Nx, Nx))
            Dxc = Dxc.tolil()
            Dxc[0, -1] = -1; Dxc[-1, 0] = 1
            Dxc /= (2*dx)
            Dyc = diags([-ex_y, ex_y], [-1, 1], shape=(Ny, Ny))
            Dyc = Dyc.tolil()
            Dyc[0, -1] = -1; Dyc[-1, 0] = 1
            Dyc /= (2*dy)

            # 2D operators via Kronecker products
            Ix = eye(Nx); Iy = eye(Ny)
            L_diff = kron(Iy, Dxx) + kron(Dyy, Ix)
            L_adv  = vx * kron(Iy, Dxc) + vy * kron(Dyc, Ix)

            L_op = -L_adv + D * L_diff
            Ntot = Nx * Ny

            A = eye(Ntot) - theta * dt * L_op
            B = (eye(Ntot) + (1 - theta) * dt * L_op)

            u_flat = u.reshape(-1)
            rhs = B @ u_flat
            u_new = spsolve(A.tocsr(), rhs)
            u = u_new.reshape((Nx, Ny))

        t += dt

    if show_plot:
        plt.contourf(X, Y, u_final, levels=50, cmap='viridis')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Final solution")
        plt.show()

    # Compute total mass
    total_mass = np.sum(u) * dx * dy

    return X, Y, u, total_mass

results = simulate_advection_diffusion_2d(dx=0.02, dy=0.02, dt=0.001)

print(results)

X, Y, u_final, total_mass = simulate_advection_diffusion_2d(
    Lx=1.0, Ly=1.0,
    vx=1.0, vy=0.5,
    D=0.01,
    dx=0.02, dy=0.02,    # spatial discretization hyperparameters
    dt=0.005,            # time step
    t_final=0.5,
    theta=0.5
)

print(total_mass)


# Values to try
X = np.array([[0.02, 0.2, 0.5], [0.2, 0.02, 0.01], [0.2, 0.2, 0.2]])

# Results
Y = np.array([])

# Get results
for x in X:
    _, _, _, y = simulate_advection_diffusion_2d(dx=x[0], dy=x[1], dt=x[2])
    Y = np.append(Y, y)

print(f"X = {X}")
print(f"Y = {Y}")

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

#exit()

# Compute accurate answer
acc_step_x = 0.005
acc_step_t = 0.001
_, _, _, y_accuarte = simulate_advection_diffusion_2d(dx=acc_step_x, dy=acc_step_x, dt=acc_step_t)

print(f"\nAccurate prediction using dx/dy step {acc_step_x} and dt step {acc_step_t} gives f(0) = {y_accuarte}\n")

diff = abs(out['mu'][0] - y_accuarte)
print(f"This is a difference of {diff:.4f}")


