import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'font.size': 14})

# Constants
RT = 8.314 * 298
F = 96485.0
cmax = 7.5 * 10e-10
Ava = 6.022e23
eV_to_J = 1.60218e-19
k_V = cmax * 10**1
k_T = cmax * 10**-2
partialPH2 = 1
beta = 0.5
scanrate = 0.025
max_time = 240
t = np.arange(0.0, max_time, scanrate)
endtime = t[-1]
duration = [0, endtime]
thetaA_H0 = 0.99
thetaA_Star0 = 1.0 - thetaA_H0
theta0 = np.array([thetaA_Star0, thetaA_H0])

# Potential function
def potential(x):
    return -0.2

# Model function for a given GHad in eV
def run_model(GHad_eV):
    GHad = GHad_eV * Ava * eV_to_J

    def eqpot(theta):
        thetaA_star, thetaA_H = theta
        return (-GHad/F) + (RT * np.log(thetaA_star / thetaA_H)) / F

    def rates_r0(t, theta):
        thetaA_star, thetaA_H = theta
        V = potential(t)
        U_V = eqpot(theta)
        r_V = k_V * (thetaA_star ** (1 - beta)) * (thetaA_H ** beta) * np.exp(beta * GHad / RT) * (np.exp(-(beta) * F * (V - U_V) / RT) - np.exp((1 - beta) * F * (V - U_V) / RT))
        r_T = k_T * ((thetaA_H ** 2) - (partialPH2 * (thetaA_star ** 2) * np.exp((-2 * GHad) / RT)))
        return r_V, r_T

    def sitebal_r0(t, theta):
        r_V, r_T = rates_r0(t, theta)
        dtheta_star = ((-r_V) + 2*r_T) / cmax
        dtheta_H = ((r_V) - (2*r_T)) / cmax
        return [dtheta_star, dtheta_H]

    soln = solve_ivp(sitebal_r0, duration, theta0, t_eval=t, method='BDF')
    thetaA_H = soln.y[1, :]
    r0_vals = np.array([rates_r0(time, theta) for time, theta in zip(t, soln.y.T)])
    volmer_rate = r0_vals[:, 0]
    tafel_rate = r0_vals[:, 1]
    curr1 = volmer_rate * -F * 1000  # mA/cm²
    max_current = np.max(np.abs(curr1[100]))
    return GHad_eV, max_current

# Run for multiple GHad values
GHad_values_eV = np.linspace(-0.3, 0.2, 20)
results = [run_model(val) for val in GHad_values_eV]

# Plotting results
df = pd.DataFrame(results, columns=["GHad (eV)", "Max |Current| (mA/cm²)"])
df.plot(x="GHad (eV)", y="Max |Current| (mA/cm²)", marker='o', grid=True, title="Max Current Density vs GHad")
plt.xlabel("GHad (eV)")
plt.ylabel("Max |Current| (mA/cm²)")
plt.tight_layout()
plt.show()