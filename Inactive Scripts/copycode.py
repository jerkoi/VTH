import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Physical constants
RT = 8.314 * 298  # J/mol
F = 96485.0       # C/mol
cmax = 7.5e-10    # mol.cm-2.s-1

# Model parameters
k1 = 1e5 * cmax  # Frequency factor for hydrogen adsorption
kf = np.array([k1])  # Only considering r0
beta = np.array([0.5])  # Transfer coefficient

# Define the potential as a function of time
def potential(x):
    UpperV = 0.95
    LowerV = 0.6
    scanrate = 0.05  # Scan rate in V/s
    timescan = (UpperV - LowerV) / scanrate
    if x % (2 * timescan) < timescan:
        Vapp = LowerV + scanrate * (x % timescan)
    else:
        Vapp = UpperV - scanrate * (x % timescan)
    return Vapp

# Function to calculate the rate r0
def rates_r0(theta, V):
    thetaA_Star, thetaA_H = theta  # Unpack surface coverage
    dG0 = -RT * np.log(thetaA_H / thetaA_Star)  # Reaction free energy for r0
    U0 = -dG0 / F  # Equilibrium potential
    r0 = (kf[0] * np.exp(beta[0] * dG0 / RT) * (thetaA_H**beta[0]) * 
          (thetaA_Star**(1 - beta[0])) *
          (np.exp(-beta[0] * F * (V - U0) / RT) - 
           np.exp((1 - beta[0]) * F * (V - U0) / RT)))
    return r0

# Define the ODEs for the site balance
def sitebal_r0(theta, t):
    V = potential(t)  # Applied voltage
    r0 = rates_r0(theta, V)  # Rate for r0
    dthetadt = [-r0 / cmax, r0 / cmax]  # Change in \theta_A^* and \theta_A^H
    return dthetadt

# Initial conditions
thetaA_H0 = 0.01  # Initial coverage of Hads
thetaA_Star0 = 1.0 - thetaA_H0  # Initial coverage of empty sites
theta0 = [thetaA_Star0, thetaA_H0]

# Time array for integration
tStop = 64.0
tInc = 0.01
t = np.arange(0.0, tStop, tInc)
ts = int(tStop / tInc)

for i in range(ts):
    E[i]=potential(t[i])
    

    
# plot E(V) vs t(s)   
plt.plot(t[2:ts], E[2:ts], 'b')
plt.xlabel('Time (s)')
plt.ylabel('E vs RHE (V)')
plt.show()

# Solve the ODE
soln = odeint(sitebal_r0, theta0, t)

# Extract coverages
thetaA_Star = soln[:, 0]
thetaA_H = soln[:, 1]

# Calculate r0 over time
r0_vals = [rates_r0(theta, potential(time)) for theta, time in zip(soln, t)]

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(t, thetaA_Star, label=r'$\theta_A^*$ (empty sites)', color='magenta')
plt.plot(t, thetaA_H, label=r'$\theta_A^H$ (adsorbed hydrogen)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Coverage')
plt.legend()
plt.title('Surface Coverage vs. Time')
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(t, r0_vals, label=r'$r_0$ (rate of hydrogen adsorption)', color='green')
plt.xlabel('Time (s)')
plt.ylabel(r'$r_0$ (mol/cm²/s)')
plt.legend()
plt.title('Reaction Rate vs. Time')
plt.grid()
plt.show()
