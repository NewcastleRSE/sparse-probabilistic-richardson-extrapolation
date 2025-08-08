# Python modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d

# Application modules
from sparse_pre.extrapolation import extrapolation

# -----------------------------
# SIR model differential equations
# -----------------------------
def sir_model(t, y, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def run_sir_model(diff_tol = 1e-8, integrate_tol = 1e-6, do_plot = False):


    # -----------------------------
    # Parameters
    # -----------------------------

    N = 1000
    beta = 0.3
    gamma = 0.1
    S0 = N - 1
    I0 = 1
    R0 = 0
    y0 = [S0, I0, R0]

    t_span = (0, 160)
    t_eval = np.linspace(*t_span, 500)

    # -----------------------------
    # Solve SIR system together
    # -----------------------------
    sol = solve_ivp(
        fun=sir_model,
        t_span=t_span,
        y0=y0,
        args=(beta, gamma, N),
        t_eval=t_eval,
        method='RK45',
        atol=diff_tol  
    )

    S, I, R = sol.y
    t = sol.t

    # -----------------------------
    # Tolerance-controlled integral of I(t)
    # -----------------------------
    # Interpolate I(t) for smooth integration
    I_interp = interp1d(t, I, kind='cubic', fill_value="extrapolate")

    # Integrate I(t) using adaptive quadrature (quad)
    total_infected, err = quad(I_interp, t_span[0], t_span[1], epsabs=integrate_tol, epsrel=integrate_tol)

    if do_plot:
        # -----------------------------
        # Plot Results
        # -----------------------------
        plt.figure(figsize=(10, 6))
        plt.plot(t, S, label='Susceptible')
        plt.plot(t, I, label='Infected')
        plt.plot(t, R, label='Recovered')
        plt.xlabel('Time (days)')
        plt.ylabel('Population')
        plt.title('SIR Model with Adaptive Integral of Infected')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # Print total infected
    # -----------------------------
    #print(f"Total infected over time (adaptive integral): {total_infected:.4f}")
    #print(f"Estimated integration error: {err:.2e}")

    return total_infected 

# Values to try
X = np.array([[1e-2, 1e-2], [1e-2, 1e-4], [1e-3, 1e-3], [1e-4, 1e-2], [1e-4, 1e-4], [1e-5, 1e-4]])

# Results
Y = np.array([])

# Get results
for x in X:
    y = run_sir_model(x[0], x[1])
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

# Compute accurate answer
acc_step = 1e-8
y_accuarte = run_sir_model(1e-8, 1e-8)

print(f"\nAccurate prediction using step {acc_step} gives f(0) = {y_accuarte:.4f}\n")

diff = abs(out['mu'][0] - y_accuarte)
print(f"This is a difference of {diff:.4f}")

