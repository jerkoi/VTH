#Importing libraries- numpy, odeint, error function, matplotlib
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erf
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

# Initial conditions should have exactly 4 elements in this order:
thetaA_Star0 = ...  # empty sites
thetaA_H0 = ...     # H coverage
thetaA_OH0 = ...    # OH coverage
thetaA_O0 = ...     # O coverage

theta0 = [thetaA_Star0, thetaA_H0, thetaA_OH0, thetaA_O0]

# Linear sweep voltammetrty- defining a potential as a function of time
def potential(x):
    x = np.atleast_1d(x).astype(float)
    UpperV = 0.95
    LowerV = 0.6
    scanrate = 0.05  # V/s
    timescan = (UpperV - LowerV) / scanrate
    modtime = x % (2 * timescan)

    Vapp = np.where(
        modtime < timescan,
        LowerV + scanrate * modtime,
        UpperV - scanrate * (modtime - timescan)
    )

    return Vapp[0] if Vapp.size == 1 else Vapp


# Defining free energies of adsorption and programming adsorption isotherms
def Gad(theta):
    thetaA_Star, thetaA_H, thetaA_OH, thetaA_O = theta
    Gads0=np.array([0, -0.35, 0.775, 1.83]) #adsorption free energies for [free sitesA, Had , OHad, Oad]
    Gads=F*Gads0
    GadA_OH = Gads[2] - 1000*erf(10*(thetaA_OH-0.7))*thetaA_OH # frumkin isotherm with error function
    Gads[2]=GadA_OH
    
    GadA_H = Gads[1] + 25000*thetaA_H #linear tempkin isotherm
    Gads[1]=GadA_H
    return Gads

# Function to calculate reaction free energies from adsorption free energies
def Grxn(Gads):
    GadA_Star, GadA_H, GadA_OH, GadA_O = Gads
    dGr0 = GadA_H - GadA_Star
    dGr1 = GadA_Star - GadA_OH
    dGr2 = GadA_OH - GadA_O
    
    Grxns = np.array([dGr0, dGr1, dGr2])
    return Grxns

# Function to calculate U and Keq from theta, dG
def eqpot(theta):
    thetaA_Star, thetaA_H, thetaA_OH, thetaA_O = theta # unpack surface coverage
    Gad1 = Gad(theta)
    dG = Grxn(Gad1)
    U0 = -dG[0]/F + (RT*np.log(thetaA_Star/thetaA_H))/F
    U1 = -dG[1]/F + (RT*np.log(thetaA_OH/thetaA_Star))/F
    U2 = -dG[2]/F + (RT*np.log(thetaA_O/thetaA_OH))/F
    eqconst =[U0, U1, U2]
    return eqconst



# Function to calculate elementary reaction rates from theta, V, U
def rates(theta, V):
    
    thetaA_Star, thetaA_H, thetaA_OH, thetaA_O = theta # unpack surface coverage
    
    Gad1 = Gad(theta) #adsorption free energies (function of coverage)
    
    dG = Grxn(Gad1) #reaction free energies (function of adsorption free energies)
    
    U0, U1, U2 = eqpot(theta)  # call function to find U for given theta
   
    
    r0 = (kf[0]*np.exp(beta[0]*dG[0]/RT)*(thetaA_H**beta[0])*(thetaA_Star**(1-beta[0]))*
         (np.exp(-beta[0]*F*(V-U0)/RT)-np.exp((1-beta[0])*F*(V-U0)/RT)))
    
    r1 = (kf[1]*np.exp(beta[1]*dG[1]/RT)*(thetaA_Star**beta[1])*(thetaA_OH**(1-beta[1]))*
         (np.exp(-beta[1]*F*(V-U1)/RT)-np.exp((1-beta[1])*F*(V-U1)/RT)))
    
    r2 = (kf[2]*np.exp(beta[2]*dG[2]/RT)*(thetaA_OH**beta[2])*(thetaA_O**(1-beta[2]))*
         (np.exp(-beta[2]*F*(V-U2)/RT)-np.exp((1-beta[2])*F*(V-U2)/RT)))
    
    
    r = [r0, r1, r2]
    return r

# Function to calculate site balances from elementary reaction steps - this is the ODE
def sitebal(time, theta, oparams):
    V = potential(time)
    r0, r1, r2 = rates(theta, V)
    dthetadt = [(r1 - r0) / cmax, r0 / cmax, (r2 - r1) / cmax, -r2 / cmax]
    return dthetadt



# Physical parameters
RT = 8.314*298   # J/mol
F = 96485.0 # C/mol
cmax= 7.5e-10 # mol.cm-2.s-1


# model parameters for fitting
k1 = 1e5*cmax # for Hydrogen adsorption
k2 = 2e9*cmax # for OH and O adsorption desorption
kf=np.array([0*k1, 1*k2, 1*k2])  # k1 and k2 are frequency factors = primary fitting parameters
beta = np.array([0.5, 0.5, 0.5])  # transfer coefficients


# Initial values - will require some thought for figuring out
thetaA_H0 = 0.01 # initial coverage Hads
thetaA_O0 = 0.1 # initial coverage Oads
thetaA_OH0 = 0.1# initial coverage OHads
thetaA_Star0 = 1.0 - (thetaA_H0 + thetaA_O0 + thetaA_OH0) # initial coverage empty sites

# Bundle parameters for ODE solver 
oparams = (beta) # this is all for now, could set up so that beta/K/dG are passed to model
theta0 = [thetaA_Star0, thetaA_H0, thetaA_OH0, thetaA_O0]  #pack into list of initial cond 

# Make time array for solution in seconds
tStop = 64.
tInc = 0.01
ts =int(tStop/tInc)
t = np.linspace(0., tStop, ts)


# defining empty arrays for potential, kinetic and total current density
E = np.empty(ts, dtype=object)
curr1 = np.empty(ts, dtype=object)
tcurr1= np.empty(ts, dtype=object) 

# define diffusion limited current, calculated from levich equation for 1600 rpm O2, 0.196 cm2 Pt(111) disk in 0.1 M HClO4
iD = -6.12 # mA/cm2

for i in range(ts):
    E[i]=potential(t[i])
    

    
# plot E(V) vs t(s)   
plt.plot(t[2:ts], E[2:ts], 'b')
plt.xlabel('Time (s)')
plt.ylabel('E vs RHE (V)')
plt.show()


# Call the ODE solver
soln1 = solve_ivp(sitebal, (t[0], t[-1]), theta0, t_eval=t, method = 'LSODA' , args=(oparams,), atol = 1e-9, rtol = 1e-6)


# plot coverage as a function of time
plt.plot(soln1.t, soln1.y[0], 'm', label='*')
plt.plot(soln1.t, soln1.y[1], 'r', label='H')
plt.plot(soln1.t, soln1.y[2], 'b', label='OH')
plt.plot(soln1.t, soln1.y[3], 'g', label='O')


plt.legend(loc='best')
plt.xlabel('t(s)')
plt.ylabel('Coverages')
plt.show()

# plot coverages on *A sites as a function of potential
plt.plot(E[:len(soln1.t)], soln1.y[0], 'm', label='*')
plt.plot(E[:len(soln1.t)], soln1.y[2], 'b', label='OH')
plt.plot(E[:len(soln1.t)], soln1.y[3], 'g', label='O')





plt.legend(loc='best')
plt.xlabel('E vs RHE (V)')
plt.ylabel('Coverages')
plt.show()


for i in range(soln1.y.shape[1]):
    theta_temp = soln1.y[:, i]  # ← each column is a [θ*, θH, θOH, θO] at time t[i]
    Vapp = potential(t[i])
    R = rates(theta_temp, Vapp)
    curr1[i] = -1000 * F * (R[0] + R[1] + R[2])


# plot kinetic current desnity as a function of potential
plt.plot(E[10:20000], curr1[10:20000], 'b')
plt.xlabel('E vs RHE(V)')
plt.ylabel('Kinetic current (mA/cm2)')
plt.show()