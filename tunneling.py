import numpy as np
import utils
import solver
import potentials

#====================== Defining the relevant parameters =====================
L = 100
Nx, Nt = (4*L, 10000)
x0 = -L/3
kappa, sigma = (500/L, L/20)
tmax = 100
mid = 0
width = L/50
V0 = 10
V = potentials.tunneling(Nx, L, mid, width, V0)
#=============================================================================

#================== Solve the problem using Crank-Nicholson ==================
isw = solver.crank_nicholson(Nx=Nx, 
                             Nt=Nt, 
                             kappa=kappa, 
                             sigma=sigma, 
                             L=L, x0=x0, 
                             tmax=tmax, 
                             V=V)
#=============================================================================

#============================== Plot and Animate =============================
utils.animDensity(isw["x"], isw["psi"], isw["V"])
utils.animRealImag(isw["x"], isw["psi"], isw["V"])
utils.plotExpectedPosition(isw["x"], isw["t"], isw["psi"])
utils.plotTimeEvolution(isw["x"], isw["t"], isw["psi"])
utils.plotSome(isw["x"], isw["t"], isw["psi"])
#=============================================================================