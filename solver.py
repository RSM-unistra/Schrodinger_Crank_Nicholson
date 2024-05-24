import numpy as np
import tqdm

# Generate a Gaussian wave packet
def gaussian_packet(x, x0, kappa, sigma, norm=True):
    a = 1 / (sigma * np.sqrt(2 * np.pi))
    b = np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    c = np.exp(1j * kappa * (x - x0))
    packet = a * b * c
    # Normalize wave packet if set to True
    if norm:
        normalization = np.sqrt(1 / np.trapz(np.abs(packet) ** 2, x=x))
        return normalization * packet
    else:
        return packet
    
# Calculate matrices A and B for Crank-Nicholson
def get_ab(Nx, dt, dx, V):
    alpha = 1j * dt / (2 * dx ** 2)
    beta = 1j * dt / 2
    A = np.zeros((Nx, Nx), dtype=complex)
    B = np.zeros((Nx, Nx), dtype=complex)
    for i in range(Nx):
        A[i, i] = 1 + alpha + beta * V[i]
        B[i, i] = 1 - alpha - beta * V[i]
        if i > 0:
            A[i, i - 1] = -alpha / 2
            B[i, i - 1] = alpha / 2
        if i < Nx - 1:
            A[i, i + 1] = -alpha / 2
            B[i, i + 1] = alpha / 2
    return A, B

# Solve the problem by applying Crank-Nicholson
def crank_nicholson(Nx, Nt, kappa, sigma, L, x0, tmax=10, V=None):
    """
    Nx: Number of spatial grid points
    Nt: Number of time steps
    kappa, sigma: Parameters for Gaussian wave packet
    L: Length of the spatial domain
    x0: Initial position of the wave packet
    tmax: Maximum time for simulation
    V: Potential function (default is ISW)
    """
    x, dx = np.linspace(-L/2, L/2, Nx, retstep=True)
    t, dt = np.linspace(0, tmax, Nt, retstep=True)
    
    # Initialize wavepacket at t=0 using Gaussian function
    psi0 = gaussian_packet(x, x0, kappa, sigma, norm=True)
    
    # No potential is set by default (Infinite Square Well)
    if V is None:
        V = np.zeros(len(x))

    # Calculate matrices A and B for time evolution
    A, B = get_ab(Nx, dt, dx, V)
    
    # Initialize array to store wavefunction psi
    psi = np.zeros((Nx, Nt), dtype=complex)
    psi[:, 0] = psi0  # Assign initial wavefunction at t=0
    
    # Calculate inverse of matrix A for time evolution
    invA = np.linalg.inv(A)
    
    # Time evolution loop
    for time in tqdm.tqdm(range(Nt - 1), desc="Computing Crank-Nicholson"):
        Y = B.dot(psi[:, time])  # Calculate Y vector for time step
        psi[:, time + 1] = invA.dot(Y)  # Update wavefunction for next time step
    
    return {"x":x, "t":t, "psi":psi, "V":V}