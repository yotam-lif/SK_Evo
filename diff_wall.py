import numpy as np
import matplotlib.pyplot as plt

# Parameters
D = 0.01  # Diffusion coefficient
L = 2.0  # Length of the domain
Nx = 400  # Number of spatial points
dx = L / (Nx - 1)  # Spatial step size
x = np.linspace(0, L, Nx)
flux = 0.1
degredation = 0.2

T = 4.0  # Total time
Nt = 4000  # Number of time steps
dt = T / Nt  # Time step size

# Stability condition for explicit scheme
# dt < dx^2 / (2D)
if dt > dx ** 2 / (2 * D):
    print("Warning: Time step is too large for stability. Reduce dt or increase Nx.")
    # Optionally, adjust Nt to satisfy stability
    print(dt, dx ** 2 / (2 * D))

# Initial condition: Gaussian centered at x0 with small width
x0 = 0.1  # Initial mean
sigma = 0.0001  # Initial standard deviation
u_initial = np.exp(- (x - x0) ** 2 / (2 * sigma ** 2))
u_initial /= (sigma * np.sqrt(2 * np.pi))  # Normalize

u = u_initial.copy()

# Time points to plot
plot_times = [0.3 * T, 0.5 * T, 0.7 * T, T]
plot_steps = [int(t / dt) for t in plot_times]

# Time-stepping loop
for n in range(1, Nt + 1):
    u_new = u.copy()

    # Update interior points
    u_new[1:-1] = u[1:-1] + D * dt / dx ** 2 * (u[2:] - 2 * u[1:-1] + u[:-2])

    # Reflecting boundary condition at x=0 (Neumann BC)
    #
    u_new[0] += flux - degredation * u_new[0]


    # No need for boundary condition at x=L (assuming u=0 or natural boundary)
    # Here, we assume natural boundary with zero second derivative at x=L
    # Alternatively, implement u[-1] = u[-2] for Neumann at x=L

    u = u_new.copy()

    # Plot at specified times
    if n in plot_steps:
        current_time = n * dt
        plt.plot(x, u, label=f't={current_time:.2f}')

# Finalize plot
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.yscale('log')
plt.title('Diffusion with Reflecting Boundary at x=0')
plt.legend()
plt.grid(True)
plt.show()
