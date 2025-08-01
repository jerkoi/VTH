#Currently working on generate_time_grid function
#found that changing the base length or period would scale the amount of curr_V values that would be evaluated
#need to make a function that can make a time array


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def generate_time_grid(period, max_time=10, scanrate=0.025, base_length=400, base_period=0.5):
    # Resolution scales inversely with period
    scale = base_period / period
    length = int(base_length * scale)
    return np.linspace(0, max_time, length)

# Physical Constants
RT = 8.314*298 #ideal gas law times temperature
F = 96485.0 #Faraday constant, C/mol
cmax = 7.5*10e-10 #mol*cm-2*s-1

# Model Parameters
Avo = 6.022*10**23
conversion_factor = 1.60218e-19  # Conversion factor from eV to J
k_V = cmax * 10**3.8
k_T = cmax * 10**8
partialPH2 = 1
beta = 0.28
#GHad = -0.3 * Avo * conversion_factor  # Convert GHad from eV to J


## dG values
dGmin_eV = -0.3 #eV
dGmax_eV = 0.3

# # potential sweep & time 
period = 0.5 #seconds
UpperV = 0.05
LowerV = -0.5
scanrate = 0.025  #scan rate in V/s
timescan = (UpperV-LowerV)/(scanrate)
max_time = 10 #seconds
t = generate_time_grid(period, base_length=100, base_period=0.5)
duration = [0, t[-1]]
endtime = t[-1]
time_index = [t]



#Initial conditions
thetaA_H0 = 0.99  # Initial coverage of Hads, needs to be high as this is reduction forward
thetaA_Star0 = 1.0 - thetaA_H0  # Initial coverage of empty sites
theta0 = np.array([thetaA_Star0, thetaA_H0])

# === Prompt User ===
print("Choose which simulations to run:")
do_static_volcano = input("Run static volcano plot? (y/n): ").strip().lower() == 'y'
do_dynamic_ghad = input("Run dynamic GHad(t) simulation? (y/n): ").strip().lower() == 'y'

# === Prepare dynamic overlay variables ===
dynamic_overlay_points = []

# === DYNAMIC GHad(t) SIMULATION ===
if do_dynamic_ghad:
    print("\nRunning dynamic GHad(t) simulation...")

    # Time-varying GHad values (in J)
    dGmin = dGmin_eV * Avo * conversion_factor
    dGmax = dGmax_eV * Avo * conversion_factor

    def dGvt(t):
        return dGmin if (t // period) % 2 == 0 else dGmax

    def potential(t): return -0.1

    def eqpot(theta, GHad):
        theta = np.asarray(theta)
        thetaA_star, thetaA_H = theta # unpack surface coverage
        ##Volmer
        U_V = (-GHad/F) + (RT*np.log(thetaA_star/thetaA_H))/F 
        #U relies on the free energy of hydrogen adsorption plus the log of surface coverage (considered a concentration)
        
        return U_V
        

    #reduction is FORWARD, oxidation is REVERSE, all variables are consistent with this
    def rates_r0(t, theta):
        GHad = dGvt(t)
        theta = np.asarray(theta)
        thetaA_star, thetaA_H = theta #surface coverages again, acting as concentrations
        V = potential(t)  # Use t directly (scalar)
        U_V = eqpot(theta, GHad) #call function to find U for given theta
        
        ##Volmer Rate Equation
        r_V = k_V * (thetaA_star ** (1 - beta)) * (thetaA_H ** beta) * np.exp(beta * GHad / RT) * (np.exp(-(beta) * F * (V - U_V) / RT) - np.exp((1 - beta) * F * (V - U_V) / RT))
        
        r_T = k_T * ((thetaA_H **2) - (partialPH2 * (thetaA_star ** 2) * np.exp((-2*GHad) / RT)))
        
        return r_V, r_T

    def sitebal(t, theta):
        r_V, r_T = rates_r0(t, theta)
        dtheta_star_dt = (-r_V + 2 * r_T) / cmax
        dtheta_H_dt = (r_V - 2 * r_T) / cmax
        return [dtheta_star_dt, dtheta_H_dt]

    soln = solve_ivp(sitebal, duration, theta0, t_eval= t, method='BDF')
    print("t_evals shape:", t.shape)
    print("soln.t shape:", soln.t.shape)
    print("soln.success:", soln.success)
    print("soln.message:", soln.message)
    r0_vals = np.array([rates_r0(time, theta) for time, theta in zip(t, soln.y.T)])
    r_V_vals = r0_vals[:, 0]
    r_T_vals = r0_vals[:, 1]
    curr_dynamic = r_V_vals * -F * 1000  # mA/cm²
    GHad_t_J = np.array([dGvt(time) for time in t])
    GHad_t_eV = GHad_t_J / (Avo * conversion_factor)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, curr_dynamic, label='Volmer Current')
    plt.ylabel("Current Density (mA/cm²)")
    plt.title("Dynamic GHad(t): Current vs Time, $k_V$ = {:.2e}, $k_T$ = {:.2e}, $beta$ = {:.2f}".format(k_V / cmax, k_T / cmax, beta))
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, GHad_t_eV)
    plt.ylabel("GHad (eV)")
    plt.xlabel("Time (s)")
    plt.title("Dynamic GHad(t): GHad vs Time")
    plt.tight_layout()
    plt.show()

    # Mask-based max current extraction
    mask_min = np.isclose(GHad_t_eV, dGmin_eV)
    mask_max = np.isclose(GHad_t_eV, dGmax_eV)
    max_curr_min = np.max(np.abs(curr_dynamic[mask_min]))
    max_curr_max = np.max(np.abs(curr_dynamic[mask_max]))
    dynamic_overlay_points.append((dGmin_eV, max_curr_min))
    dynamic_overlay_points.append((dGmax_eV, max_curr_max))

    print(f"\nMax |Current| at GHad = {dGmin_eV:.2f} eV: {max_curr_min:.3f} mA/cm²")
    print(f"Max |Current| at GHad = {dGmax_eV:.2f} eV: {max_curr_max:.3f} mA/cm²")

# === STATIC VOLCANO PLOT ===
if do_static_volcano:
    print("\nRunning static volcano plot...")

    GHad_eV_list = np.linspace(dGmin_eV, dGmax_eV, 25)
    GHad_J_list = GHad_eV_list * Avo * conversion_factor
    GHad_results = []

    for GHad, GHad_eV in zip(GHad_J_list, GHad_eV_list):
        def potential(t): return -0.1

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

        soln = solve_ivp(sitebal, duration, theta0, t_eval= t, method='BDF')
        r0_vals = np.array([rates_r0(time, theta) for time, theta in zip(t, soln.y.T)])
        curr_static = r0_vals[:, 0] * -F * 1000  # mA/cm²
        max_current = np.abs(curr_static[100])
        GHad_results.append((GHad_eV, max_current))

    # Volcano plot
    GHad_vals, abs_currents = zip(*GHad_results)
    plt.figure(figsize=(8, 5))
    plt.plot(GHad_vals, abs_currents, marker='o', label='Static GHad Scan')

    if dynamic_overlay_points:
        for ghad_val, curr_val in dynamic_overlay_points:
            plt.scatter([ghad_val], [curr_val], color='red', marker='x', s=100, label=f'Dynamic Max @ {ghad_val:.2f} eV and {curr_val:.2f} mA/cm²')
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        plt.legend(unique.values(), unique.keys())

    # plt.xlabel("GHad (eV)")
    # plt.ylabel("Max |Current Density| (mA/cm²)")
    # plt.title(f"Volcano Plot: Max Current vs GHad, $k_V$ ={k_V / cmax:.2e}, $k_T$ = {k_T / cmax:.2e}, $beta$ = {beta}, $V$ = {potential(t)}")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    max_index = np.argmax(abs_currents)
    print(f"\nMax Current (static): {abs_currents[max_index]:.3f} mA/cm² at GHad = {GHad_vals[max_index]:.3f} eV")
    print("\nStatic Volcano Summary:")
    for g, c in GHad_results:
        print(f"GHad = {g:.3f} eV → Max |Current| = {c:.3f} mA/cm²")
