import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({'font.size': 14})

###########################################################################################################################
####################################################### PARAMETERS ########################################################
###########################################################################################################################

# Physical Constants
RT = 8.314 * 298  # J/mol
F = 96485.0  # C/mol
cmax = 7.5e-9  # mol*cm^-2*s^-1
beta = 0.5

# Model Parameters
k_V = cmax * 10**-4  # Pre-exponential factor (s^-1)
alpha = 1.0         # Barrier shape factor (controls volcano width)
GHad_opt = 0.0      # Optimal GHad (eV) where activity is maximum
eta = 0.2           # Fixed overpotential (V)
partialPH2 = 1.0  # Partial pressure of H2 (atm)

# Time sweep
scanrate = 0.025  # V/s
max_time = 240
t = np.arange(0.0, max_time, scanrate)
duration = [0, t[-1]]

# Initial conditions
thetaA_H0 = 0.99
thetaA_Star0 = 1.0 - thetaA_H0
theta0 = np.array([thetaA_Star0, thetaA_H0])

############################################################################################################################
########################################################## FUNCTIONS #######################################################
############################################################################################################################

def eqpot(theta, GHad):
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta
    thetaA_star = max(thetaA_star, 1e-12)
    thetaA_H = max(thetaA_H, 1e-12)
    U_V = (-GHad / F) + (RT * np.log(thetaA_star / thetaA_H)) / F
    return U_V

def potential(t, theta, GHad):
    U_eq = eqpot(theta, GHad)
    return U_eq  # Maintain fixed overpotential η

def rates_r0(t, theta, GHad):
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta
    thetaA_star = np.clip(thetaA_star, 1e-12, 1.0)
    thetaA_H = np.clip(thetaA_H, 1e-12, 1.0)

    delta_G = GHad - GHad_opt
    activation_barrier = alpha * (delta_G ** 2)

    # Forward and backward components
    forward = thetaA_star * np.exp( -(activation_barrier - beta * F * eta) / RT )
    backward = thetaA_H * np.exp( -(activation_barrier + (1-beta) * F * eta) / RT )

    r_V = k_V * (forward - backward)

    # Tafel term (stays same)
    exp_neg2_GHad_over_RT = np.exp(-2 * GHad * F / RT)
    r_T = k_V * ((thetaA_H ** 2) - (partialPH2 * (thetaA_star ** 2) * exp_neg2_GHad_over_RT))

    return r_V, r_T


def sitebal_r0(t, theta, GHad):
    r_V, r_T = rates_r0(t, theta, GHad)
    thetaStar_rate_VT = ((-r_V) + 2 * r_T) / cmax
    thetaH_rate_VT = ((r_V) - 2 * r_T) / cmax
    return [thetaStar_rate_VT, thetaH_rate_VT]

############################################################################################################################
########################################################## SOLVER ##########################################################
############################################################################################################################

# Prepare to store results
currents = []
GHad_results = []

# List of GHad values to simulate
GHad_list = np.linspace(-2, 2, 41)  # finer points for smooth volcano

for GHad in GHad_list:
    print(f"Simulating for GHad = {GHad:.3f} eV")

    # Solve system
    soln = solve_ivp(lambda t, y: sitebal_r0(t, y, GHad), duration, theta0, t_eval=t, method='BDF')
    
    thetaA_Star = soln.y[0, :]
    thetaA_H = soln.y[1, :]

    # Recalculate rates
    r0_vals = np.array([rates_r0(time, theta, GHad) for time, theta in zip(t, soln.y.T)])
    curr1 = r0_vals[:, 0] * -F * 1000  # current from Volmer step (mA/cm²)

    # Record max absolute current
    max_current = np.max(np.abs(curr1))
    GHad_results.append((GHad, max_current))

############################################################################################################################
########################################################## PLOTS ###########################################################
############################################################################################################################

# Unpack results
GHad_vals, abs_currents = zip(*GHad_results)

# Plot Max Current vs GHad (linear)
plt.figure(figsize=(10, 6))
plt.plot(GHad_vals, abs_currents, marker='o')
plt.xlabel("GHad (eV)")
plt.ylabel("Max |Current Density| (mA/cm²)")
plt.title("Sabatier Volcano (Linear Scale)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Max Current vs GHad (log scale)
plt.figure(figsize=(10, 6))
plt.semilogy(GHad_vals, abs_currents, marker='o')
plt.xlabel("GHad (eV)")
plt.ylabel("Max |Current Density| (mA/cm²)")
plt.title("Sabatier Volcano (Log Scale)")
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# Tabulate results
df = pd.DataFrame(GHad_results, columns=["GHad (eV)", "Max |Current| (mA/cm²)"])
print(df.to_string(index=False))
