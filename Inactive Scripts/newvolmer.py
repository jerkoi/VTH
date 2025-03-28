#Importing libraries- numpy, odeint, error function, matplotlib
import numpy as np
from scipy.integrate import odeint
from scipy.special import erf
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

# Physical Constants
RT = 8.314*298 #ideal gas law times temperature
F = 96485 #Faraday constant, C/mol
cmax = 7.5e-10 #mol*cm-2*s-1

# Model Parameters
k1 = 1e5*cmax
kf = np.array([k1])
beta = np.array([0.5])

# Linear sweep voltammetry- defining a potential as a function of time
def potential(x):
    UpperV = 9
    LowerV = 0.65
    scanrate = 0.05 #scan rate in V/s
    timescan=(UpperV-LowerV)/(scanrate)
    if x%(2*timescan)<timescan:
            Vapp = LowerV + scanrate*(x%((UpperV-LowerV)/(scanrate)))
    else:
            Vapp = UpperV - scanrate*(x%((UpperV-LowerV)/(scanrate)))
    return Vapp



def rates(theta, V):
       thetaA_star, thetaA_H = theta
       dG0 = -RT * np.log(thetaA_H/thetaA_star)
       U0 = dG0 / F
       r0 = kf[0]**(1-beta[0])*k1**(beta[0])*(1-thetaA_H)**(1-beta[0])*(thetaA_H**beta[0])*(aH)**(1-beta[0])[np.exp((-beta[0]*F*(V-U0))/RT)-np.exp(((1-beta[0])*F*(V-U0))/RT)]
       return r0
