import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants (only defined once)
RT = 8.314 * 298  # J/mol
F = 96485.0       # C/mol
cmax = 7.5e-9     # mol/cm²
conversion_factor = 1.60218e-19  # eV to J
AvoNum = 6.02e23  # 1/mol
partialPH2 = 1.0
beta = 0.28
k_V = cmax * 10**3.8
k_T = cmax * 10**6
scanrate = 0.025
max_time = 240
t = np.arange(0.0, max_time, scanrate)
duration = [0, t[-1]]

# Initial coverage
thetaA_H0 = 0.99
thetaA_Star0 = 1.0 - thetaA_H0
theta0 = np.array([thetaA_Star0, thetaA_H0])

# GHad sweep values
GHad_eV_list = np.linspace(-0.3, 0.3, 25)
GHad_J_list = GHad_eV_list * AvoNum * conversion_factor

thetaH_list = []

# Store results
GHad_results = []

# Main loop: everything inside
for GHad, GHad_eV in zip(GHad_J_list, GHad_eV_list):
    print(f"\nSimulating for GHad = {GHad_eV:.2f} eV")

    # Redefine everything for this GHad value

    def potential(x):
        return -0.3

    def eqpot(theta):
        theta_star, theta_H = theta
        return (-GHad / F) + (RT * np.log(theta_star / theta_H)) / F

    def rates_r0(t, theta):
        theta_star, theta_H = theta
        V = potential(t)
        U_V = eqpot(theta)
        
        exp_beta = np.exp(beta * GHad / RT)
        exp_neg2 = np.exp(-2 * GHad / RT)

        r_V = k_V * (theta_star ** (1 - beta)) * (theta_H ** beta) * exp_beta * (
            np.exp(-beta * F * (V - U_V) / RT) - np.exp((1 - beta) * F * (V - U_V) / RT)
        )
        
        r_T = k_T * (theta_H ** 2 - partialPH2 * (theta_star ** 2) * exp_neg2)
        return r_V, r_T

    def sitebal(t, theta):
        r_V, r_T = rates_r0(t, theta)
        dtheta_star_dt = (-r_V + 2 * r_T) / cmax
        dtheta_H_dt = (r_V - 2 * r_T) / cmax
        return [dtheta_star_dt, dtheta_H_dt]

    # Run solver
    soln = solve_ivp(sitebal, duration, theta0, t_eval=t, method='BDF')
    theta_star = soln.y[0, :]
    theta_H = soln.y[1, :]

    thetaH_list.append(theta_H)

    # Calculate current from Volmer step
    r0_vals = np.array([rates_r0(time, theta) for time, theta in zip(t, soln.y.T)])
    curr1 = r0_vals[:, 0] * -F * 1000  # mA/cm²

    max_current = np.abs(curr1[100])
    GHad_results.append((GHad_eV, max_current))

# Plot results
GHad_vals, abs_currents = zip(*GHad_results)

plt.figure(figsize=(10, 6))
plt.plot(GHad_vals, abs_currents, marker='o')
plt.xlabel("GHad (eV)")
plt.ylabel("Max |Current Density| (mA/cm²)")
plt.title(f"Max Current Density vs GHad, $k_V$ ={k_V / cmax}  and $beta$ = {beta}, $V$ = {potential(t)}")
plt.grid(True)
plt.tight_layout()
plt.show()

# Convert thetaH_list to a NumPy array for easy slicing
thetaH_array_long = np.array(thetaH_list)
thetaH_array = thetaH_array_long[:, 1:]  # Shape: (25, 9600)

# Plot settings
plt.figure(figsize=(12, 6))
cmap = plt.get_cmap('viridis', len(GHad_eV_list))  # Create a colormap with 25 colors

# Plot each theta_H vs time for each GHad
for i, GHad_val in enumerate(GHad_eV_list):
    plt.plot(t[1:], thetaH_array[i], label=f'{GHad_val:.2f} eV', color=cmap(i))

# Labeling
plt.xlabel('Time (s)')
plt.ylabel(r'$\theta_H$ Coverage')
plt.title(r'$\theta_H$ vs Time for Different $G_{\mathrm{Had}}$ Values')
plt.legend(title='GHad (eV)', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()


# Print best result
max_index = np.argmax(abs_currents)
print(f"\nMax Current: {abs_currents[max_index]:.3f} mA/cm² at GHad = {GHad_vals[max_index]:.3f} eV")

print("\nSummary of GHad vs Max Current:")
for g, c in GHad_results:
    print(f"GHad = {g:.3f} eV → Max |Current| = {c:.3f} mA/cm²")
