import numpy as np
from scipy.integrate import odeint
from scipy.special import erf
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

# Physical Constants
RT = 8.314 * 298  # Ideal gas law times temperature
F = 96485.0  # Faraday constant, C/mol
cmax = 7.5e-10  # mol*cm-2*s-1

# Model Parameters
A = 1e3
k1 = A * cmax
beta = 0.5
GHad = F * -0.35  # Free energy of hydrogen adsorption


# Initial conditions
thetaA_H0 = 0.99  # Initial coverage of Hads, needs to be high as this is reduction forward
thetaA_Star0 = 1.0 - thetaA_H0  # Initial coverage of empty sites
theta0 = [thetaA_Star0, thetaA_H0]

# Time bounds for integration
tStop = 60.0
tInc = 0.01
t = np.arange(0.0, tStop, tInc)
ts = int(tStop / tInc)

# Linear sweep voltammetry - defining a potential as a function of time
def potential(t):
    UpperV = 0.95
    LowerV = 0.6
    scanrate = 0.05  # scan rate in V/s
    timescan = (UpperV - LowerV) / (scanrate)
    if t % (2 * timescan) < timescan:
        Vapp = LowerV + scanrate * (t % ((UpperV - LowerV) / (scanrate)))
    else:
        Vapp = UpperV - scanrate * (t % ((UpperV - LowerV) / (scanrate)))
    return Vapp

# Function to calculate U and Keq from theta, dG
def eqpot(theta, V):
    thetaA_Star, thetaA_H = theta  # Unpack surface coverage
    # Ensure thetaA_Star and thetaA_H are positive and non-zero
    thetaA_Star = np.maximum(thetaA_Star, 1e-10)  # Avoid division by zero
    thetaA_H = np.maximum(thetaA_H, 1e-10)  # Avoid log of zero
    U0 = (-GHad / F) + (RT * np.log(thetaA_Star / thetaA_H)) / F
    return U0

# Function to calculate elementary reaction rates from theta, V
def rates_r0(theta, V):
    thetaA_star, thetaA_H = theta  # Surface coverages again, acting as concentrations
    # Ensure valid values for theta
    thetaA_star = np.maximum(thetaA_star, 1e-10)  # Avoid negative or zero values
    thetaA_H = np.maximum(thetaA_H, 1e-10)  # Avoid negative or zero values
    U0 = eqpot(theta, V)  # Call function to find U for given theta
    
    ex_cd = k1 * np.exp((beta * GHad) / RT) * (thetaA_star ** (1 - beta)) * (thetaA_H ** beta)
    exp = (np.exp(-beta * F * (V - U0) / RT) - np.exp((1 - beta) * F * (V - U0) / RT))
    
    r0 = ex_cd * exp
    return r0

# Function to calculate site balances from elementary reaction steps - this is the ODE
def sitebal_r0(theta, t, oparams):
    beta, k1, cmax = oparams  # Unpack the parameters
    V = potential(t)  # Calculate applied voltage from time (t)
    r0 = rates_r0(theta, V)  # Call rates function to calculate reaction rate
    dthetadt = [-r0 / cmax, r0 / cmax]  # These are the site balances
    return dthetadt


# Define initial conditions
thetaA_H0 = 0.99  # Initial coverage of Hads
thetaA_Star0 = 1.0 - thetaA_H0  # Initial coverage empty sites
theta0 = [thetaA_Star0, thetaA_H0]

# Bundle parameters for ODE solver
oparams = (beta, k1, cmax)  # Pass k1, cmax, and beta to the solver
theta0 = [thetaA_Star0, thetaA_H0]  # Pack into list of initial conditions

# Solve the ODE using odeint
soln = odeint(sitebal_r0, theta0, t, args=(oparams,))

# Extract coverages
thetaA_Star = soln[:, 0]
thetaA_H = soln[:, 1]

# Calculate r0 over time
r0_vals = [rates_r0(theta, potential(ti)) for theta, ti in zip(soln, t)]

# Calculate current density
curr1 = np.empty(ts, dtype=object)
for i in range(ts):
    theta_temp = soln[i, :]
    Vapp = potential(t[i])
    R = rates_r0(theta_temp, Vapp)
    curr1[i] = -1000 * F * R  # Total current from all reactions

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(t, thetaA_Star, label=r'$\theta_A^*$ (empty sites)', color='magenta')
plt.plot(t, thetaA_H, label=r'$\theta_A^H$ (adsorbed hydrogen)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Coverage')
plt.legend(loc='best')
plt.title('Surface Coverage vs. Time')
plt.grid()
plt.show()

# Plot kinetic current density as a function of potential
plt.plot(t, curr1, 'b')
plt.xlabel('Time (s)')
plt.ylabel('Kinetic current (mA/cm²)')
plt.title('Kinetic Current vs. Time')
plt.show()
