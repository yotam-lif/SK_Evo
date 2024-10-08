import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the temperature
T = 0.1  # Low temperature
beta = 1 / T
r_form = False

# Define the Glauber dynamics rate function
def glauber_rate(k: np.ndarray) -> np.ndarray:
    return 0.5 * (1 - np.tanh(beta * k))

def sswm_rate(k: np.ndarray) -> np.ndarray:
    k_mod = np.where(k < 0, np.abs(k), 0)
    return k_mod / np.sum(k_mod)

# Define the rate function for Glauber dynamics
def r(k: np.ndarray, glauber=True) -> np.ndarray:
    return glauber_rate(k) if glauber else sswm_rate(k)

# Define the initial conditions for P_+(k, t) and P_-(k, t)
def initial_P_plus(k: np.ndarray) -> np.ndarray:
    return (1 / (2 * np.pi)) * np.exp(-0.5 * (k ** 2))

def initial_P_minus(k: np.ndarray) -> np.ndarray:
    return np.zeros_like(k)

# Define the time-dependent diffusion coefficient D_sigma(t) using the trapezoidal rule
def D_sigma(P_sigma, k_grid):
    integrand = r(k_grid) * P_sigma
    integral = np.trapezoid(integrand, k_grid)
    return 2 * integral

# Compute drift velocity term v(t) using the trapezoidal rule
def v_t(P_plus, P_minus, k_grid):
    grad_P_plus = np.gradient(P_plus, k_grid)
    grad_P_minus = np.gradient(P_minus, k_grid)
    v_integrand = r(k_grid) * (grad_P_plus + grad_P_minus)
    v = 2 * np.trapezoid(v_integrand, k_grid)
    return v

# Define the coupled PDE system for P_+(k, t) and P_-(k, t)
def compute_dP_dt(P_plus, P_minus, k_grid):
    # Numerically compute D_+(t) and D_-(t)
    D_plus = D_sigma(P_plus, k_grid)
    D_minus = D_sigma(P_minus, k_grid)

    # Compute the drift velocity term v(t)
    v = v_t(P_plus, P_minus, k_grid)

    # Compute the gradients
    grad_P_plus = np.gradient(P_plus, k_grid)
    grad_P_minus = np.gradient(P_minus, k_grid)

    # Compute second derivatives
    d2P_plus = np.gradient(grad_P_plus, k_grid)
    d2P_minus = np.gradient(grad_P_minus, k_grid)

    # Compute dP_plus_dt and dP_minus_dt
    dP_plus_dt = (r(-k_grid, glauber=r_form) * P_minus
                 - r(k_grid, glauber=r_form) * P_plus
                 - v * grad_P_plus
                 - D_minus * d2P_plus)

    dP_minus_dt = (r(-k_grid, glauber=r_form) * P_plus
                  - r(k_grid, glauber=r_form) * P_minus
                  - v * grad_P_minus
                  - D_plus * d2P_minus)

    return dP_plus_dt, dP_minus_dt

# Implement the RK4 integration method
def solve_coupled_pde_RK4(k_grid, t_grid):
    # Initialize P_+(k, t) and P_-(k, t)
    P_plus = initial_P_plus(k_grid)
    P_minus = initial_P_minus(k_grid)

    dt = t_grid[1] - t_grid[0]

    for _ in tqdm(t_grid, desc="Solving PDE"):
        # Compute k1
        dP_plus_dt1, dP_minus_dt1 = compute_dP_dt(P_plus, P_minus, k_grid)

        # Compute k2
        P_plus_k2 = P_plus + 0.5 * dt * dP_plus_dt1
        P_minus_k2 = P_minus + 0.5 * dt * dP_minus_dt1
        dP_plus_dt2, dP_minus_dt2 = compute_dP_dt(P_plus_k2, P_minus_k2, k_grid)

        # Compute k3
        P_plus_k3 = P_plus + 0.5 * dt * dP_plus_dt2
        P_minus_k3 = P_minus + 0.5 * dt * dP_minus_dt2
        dP_plus_dt3, dP_minus_dt3 = compute_dP_dt(P_plus_k3, P_minus_k3, k_grid)

        # Compute k4
        P_plus_k4 = P_plus + dt * dP_plus_dt3
        P_minus_k4 = P_minus + dt * dP_minus_dt3
        dP_plus_dt4, dP_minus_dt4 = compute_dP_dt(P_plus_k4, P_minus_k4, k_grid)

        # Update P_plus and P_minus
        P_plus += (dt / 6.0) * (dP_plus_dt1 + 2*dP_plus_dt2 + 2*dP_plus_dt3 + dP_plus_dt4)
        P_minus += (dt / 6.0) * (dP_minus_dt1 + 2*dP_minus_dt2 + 2*dP_minus_dt3 + dP_minus_dt4)

        # Optional: Ensure numerical stability by enforcing non-negativity
        P_plus = np.maximum(P_plus, 0)
        P_minus = np.maximum(P_minus, 0)

        # Optional: Normalize the probability densities
        P_plus_norm = np.trapezoid(P_plus, k_grid)
        P_minus_norm = np.trapezoid(P_minus, k_grid)
        if P_plus_norm > 0:
            P_plus /= P_plus_norm
        if P_minus_norm > 0:
            P_minus /= P_minus_norm

    return P_plus, P_minus

# Define the main function
def main():
    # Define k and t grids
    k_border = 5.0
    k_grid = np.linspace(-k_border, k_border, 1000)  # Define k grid
    t_final = 50.0
    num_steps = 30000
    t_grid = np.linspace(0.0, t_final, num_steps)  # Define t grid

    # Solve the coupled PDE using RK4
    P_plus_solution, P_minus_solution = solve_coupled_pde_RK4(k_grid, t_grid)

    # Plot solutions
    plt.figure(figsize=(10, 6))
    plt.plot(k_grid, P_plus_solution, label='$P_+(k, t)$', color='red')
    plt.plot(k_grid, P_minus_solution, label='$P_-(k, t)$', color='blue')
    plt.xlabel('k')
    plt.ylabel('Probability Density')
    plt.title('MS Solution using RK4')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
