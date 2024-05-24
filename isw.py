import numpy as np
import utils
import solver
import potentials

#====================== Defining the relevant parameters =====================
L = 100
Nx, Nt = (4*L, 10000)
x0 = 0
kappa, sigma = (500/L, L/20)
tmax = 100
#=============================================================================

#================== Solve the problem using Crank-Nicholson ==================
isw = solver.crank_nicholson(Nx=Nx, 
                             Nt=Nt, 
                             kappa=kappa, 
                             sigma=sigma, 
                             L=L, x0=x0, 
                             tmax=tmax, 
                             V=None)
#=============================================================================

#============================== Plot and Animate =============================
utils.animDensity(isw["x"], isw["psi"], isw["V"])
#utils.animRealImag(isw["x"], isw["psi"], isw["V"])
#utils.plotExpectedPosition(isw["x"], isw["t"], isw["psi"])
#utils.plotTimeEvolution(isw["x"], isw["t"], isw["psi"])
#utils.plotSome(isw["x"], isw["t"], isw["psi"])
#=============================================================================