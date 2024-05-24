import numpy as np

def infiniteSquareWell(x):
    pot = np.zeros(len(x))
    pot[0] = 1e3
    pot[-1] = 1e3
    return pot

def tunneling(Nx, L, mid, width, V0):
    pot = np.zeros(Nx)
    x = np.linspace(-L/2, L/2, Nx)
    cond = (x >= mid - width/2) & (x <= mid + width/2)
    pot[cond] = V0
    return pot

def harmonic(Nx, L, omega):
    x = np.linspace(-L/2, L/2, Nx)
    return  omega * x**2 / 2