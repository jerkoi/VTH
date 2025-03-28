import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'font.size': 14})

# Physical Constants
RT = 8.314 * 298  # ideal gas law times temperature
F = 96485.0  # Faraday constant, C/mol
cmax = 7.5 * 10e-10  # mol*cm-2*s-1

# Model Parameters
A = 1 * 10**2
k1 = A * cmax
k_T = cmax * 10**-2
partialPH2 = 1
beta = 0.5
GHad = F * -0.15  # free energy of hydrogen adsorption
UpperV = 0.6
LowerV = 0.1
scanrate = 0.025  # scan rate in V/s
timestep = 0.01
timescan = (UpperV - LowerV) / scanrate
t = np.arange(0.0, 2 * timescan, scanrate)
endtime = t[-1]
duration = [0, endtime]

# Sawtooth Voltage Sweep Function
def potential_sawtooth(x):
    period = 2 * timescan
    half_idx = len(t) // 2
    forward_sweep = np.linspace(LowerV, UpperV, half_idx)
    reverse_sweep = np.linspace(UpperV, LowerV, len(t) - half_idx)
    Vapp = np.concatenate((forward_sweep, reverse_sweep))
    return Vapp[np.searchsorted(t, x, side="right") - 1]

# Initial conditions
thetaA_H0 = 0.99  # Initial coverage of Hads
thetaA_star0 = 1.0 - thetaA_H0  # Initial coverage of empty sites
theta0 = np.array([thetaA_star0, thetaA_H0])

# Function to calculate U and Keq from theta, dG
def eqpot(theta):
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta  # unpack surface coverage
    U0 = (-GHad / F) + (RT * np.log(thetaA_star / thetaA_H)) / F 
    return U0

# Reduction is FORWARD, oxidation is REVERSE
def rates_r0(t, theta):
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta  # surface coverages
    V = potential_sawtooth(t)  # Use new potential function
    U0 = eqpot(theta)  # call function to find U

    # Volmer equations
    r_V = k1 * (thetaA_star ** (1 - beta)) * (thetaA_H ** beta) * np.exp(beta * GHad / RT) * (np.exp(-(beta) * F * (V - U0) / RT) - np.exp((1 - beta) * F * (V - U0) / RT))
    
    # Tafel Rate
    r_T = k_T * ((thetaA_H**2) - ((thetaA_star**2) * (partialPH2) * np.exp((-2 * GHad) / RT)))
    
    return r_V, r_T

# Site balance equations
def sitebal_r0(t, theta):
    r_V, r_T = rates_r0(t, theta)
    dthetadt = [((-2 * r_V) + r_T) / cmax, ((2 * r_V) - r_T) / cmax]  # [0 = star, 1 = H]
    return dthetadt

# Solve ODEs
soln = solve_ivp(sitebal_r0, duration, theta0, t_eval=t, method='BDF')

# Extract coverages from solution
thetaA_Star = soln.y[0, :]
thetaA_H = soln.y[1, :]

# Compute reaction rates and kinetic current
r_vals = np.array([rates_r0(time, theta) for time, theta in zip(t, soln.y.T)]) 
curr1 = r_vals * -F

# Compute potential array
V = np.array([potential_sawtooth(ti) for ti in t])

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(t, V, label='Voltage (V)', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(t, thetaA_Star, label=r'$\theta_A^*$ (empty sites)', color='magenta')
plt.plot(t, thetaA_H, label=r'$\theta_A^H$ (adsorbed hydrogen)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Coverage')
plt.grid()
plt.legend()
plt.title(f'Surface Coverage vs. Time, A = {A}')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(t, r_vals[:, 0], label=r'$r_0$ (rate of hydrogen adsorption)', color='green')
plt.xlabel('Time (s)')
plt.ylabel(r'$r_0$ (mol/cm²/s)')
plt.legend()
plt.title(f'Reaction Rate vs. Time, A = {A}')
plt.grid()
plt.show()

plt.plot(V, curr1, color='g')
plt.xlabel('V vs RHE (V)')
plt.ylabel('Kinetic current (mA/cm²)')
plt.title(f'Kinetic Current vs Potential, A = {A}')
plt.grid()
plt.show()

# Export Data to Excel
data = {
    "Time (s)": t,
    "Voltage (V)": V,
    "Rate values": r_vals[:, 0],
    "ThetaA_Star": thetaA_Star,
    "ThetaA_H": thetaA_H,
}
df = pd.DataFrame(data)
df.to_excel("reaction_data.xlsx", index=False)

print("Data exported successfully to reaction_data.xlsx")
