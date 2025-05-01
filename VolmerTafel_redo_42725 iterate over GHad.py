import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'font.size': 14})

###########################################################################################################################
###########################################################################################################################
####################################################### PARAMETERS ########################################################
###########################################################################################################################
###########################################################################################################################

# Physical Constants
RT = 8.314*298 #ideal gas law times temperature
F = 96485.0 #Faraday constant, C/mol
cmax = 7.5*10e-10 #mol*cm-2

# Model Parameters
k_V = cmax * 10**2
k_T = cmax * 10**2
conversion_factor = 1.60218e-19  # Conversion factor from eV to J
AvoNum = 6.02*10**23
partialPH2 = 1
beta = 0.5
GHad = 0.3 * AvoNum * conversion_factor #J/mol

# # potential sweep & time 
UpperV = 0.5
LowerV = -0.5
scanrate = 0.025  #scan rate in V/s
timescan = (UpperV-LowerV)/(scanrate)
max_time = 240
t = np.arange(0.0, max_time, scanrate)
endtime = t[-1]
duration = [0, endtime]
time_index = [t]
rV_list = []
rT_list = []

#Initial conditions
thetaA_H0 = 0.99  # Initial coverage of Hads, needs to be high as this is reduction forward
thetaA_Star0 = 1.0 - thetaA_H0  # Initial coverage of empty sites
theta0 = np.array([thetaA_Star0, thetaA_H0])

############################################################################################################################
############################################################################################################################
########################################################## FUNCTIONS #######################################################
############################################################################################################################
############################################################################################################################




#Linear sweep voltammetry- defining a potential as a function of time
def potential(x):
    # if x%(2*timescan)<timescan:
    #     return LowerV + scanrate*(x% timescan)
    # else:   
    #     return UpperV - scanrate*((x - timescan) % timescan)
    return -0.1


#Function to calculate U and Keq from theta, dG
def eqpot(theta):
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta # unpack surface coverage
    ##Volmer
    U_V = (-GHad/F) + (RT*np.log(thetaA_star/thetaA_H))/F 
    #U relies on the free energy of hydrogen adsorption plus the log of surface coverage (considered a concentration)
    
    return U_V
    

#reduction is FORWARD, oxidation is REVERSE, all variables are consistent with this
def rates_r0(t, theta):
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta
    V = potential(t)
    U_V = eqpot(theta)
    exp_beta_GHad_over_RT = np.exp(beta * GHad / RT)
    exp_neg2_GHad_over_RT = np.exp(-2 * GHad / RT)
    
    r_V = k_V * (thetaA_star ** (1 - beta)) * (thetaA_H ** beta) * exp_beta_GHad_over_RT * (np.exp(-(beta) * F * (V - U_V) / RT) - np.exp((1 - beta) * F * (V - U_V) / RT))
    
    r_T = k_T * ((thetaA_H ** 2) - (partialPH2 * (thetaA_star ** 2) * exp_neg2_GHad_over_RT))
    
    return r_V, r_T

def sitebal_r0(t, theta):
    r_V, r_T = rates_r0(t, theta)
    thetaStar_rate_VT = ((-r_V) + 2*r_T) / cmax
    thetaH_rate_VT = ((r_V) - 2*r_T) / cmax
    dthetadt = [thetaStar_rate_VT, thetaH_rate_VT]
    return dthetadt

V = np.array([potential(ti) for ti in t])
curr1 = np.empty(len(t), dtype=object)
tcurr1= np.empty(len(t), dtype=object)

# whether a rate is positive or negative depends on the associated reaction affects each site balance
# Heyrovsky (r1) is positive in theta_star_rate because as the H reaction moves forward, the number of empty sites increases
# Volmer (r0) is negative in theta_star_rate because as the V reaction moves forward, the number of empty sites decreases
# Heyrovsky (r1) is negative in theta_H_rate because as the H reaction moves forward, the number of H-occupied sites decreases
# Volmer (r0) is positive in theta_H_rate because as the V reaction moves forward, the number of H-occupied sites increases

############################################################################################################################################################
############################################################################################################################################################
########################################################## SOLVER ##########################################################################################
############################################################################################################################################################
############################################################################################################################################################

# Prep lists to store per-k_T simulation results
currents = []
thetaA_H_list = []
thetaA_star_list = []
GHad_results = []


# # List of GHad values to try
GHad_list = [GHad]

for new_GHad in GHad_list:

    #GHad = new_GHad  # update global for this simulation
    print(f"Simulating for GHad = {GHad:.3f}")
    # Solve the system
    soln = solve_ivp(sitebal_r0, duration, theta0, t_eval=t, method='BDF')
    
    # Extract theta
    thetaA_Star = soln.y[0, :]
    thetaA_H = soln.y[1, :]

    # Recalculate rates
    r0_vals = np.array([rates_r0(time, theta) for time, theta in zip(t, soln.y.T)])
    curr1 = r0_vals[:, 0] * -F * 1000  # current from Volmer step

  # record absolute max current
    thetaH_coverage = thetaA_H[100]
    thetaStar_coverage = thetaA_Star[100]
    max_current = (np.abs(curr1[100]))
    GHad_results.append((GHad, max_current))  # save result


############################################################################################################################################################
############################################################################################################################################################
########################################################## Value Extracting ################################################################################
############################################################################################################################################################
############################################################################################################################################################

###takes only volmer rate to compute kinetic current density
volmer_rate = r0_vals[:, 0]
tafel_rate = r0_vals[:, 1]

###########################################################################################################################
###########################################################################################################################
########################################################## PLOTS ############################################################
###########################################################################################################################
###########################################################################################################################
# #Plot results
# plt.figure(figsize=(8, 6))
# plt.plot(t[1:], thetaA_Star[1:], label=r'$\theta_A^*$ (empty sites)', color='magenta')
# plt.plot(t[1:], thetaA_H[1:], label=r'$\theta_A^H$ (adsorbed hydrogen)', color='blue')
# plt.xlabel('Voltage vs. RHE (V)')
# plt.ylabel('Coverage')
# plt.grid()
# plt.legend()
# plt.title('Surface Coverage vs. Time')
# plt.show()


# # plot kinetic current desnity as a function of potential
# plt.plot(V[10:20000], curr1[10:20000], 'b')
# plt.xlabel('Voltage vs. SHE (V)')
# plt.ylabel('Kinetic current (mA/cm2)')
# plt.title(r'Kinetic Current vs Time, GHad = %.2f eV, k_V = %.2e, k_T = %.2e' % (GHad / (AvoNum * conversion_factor), (k_V / cmax), (k_T / cmax)))
# plt.grid()
# plt.show()

# Unpack results
GHad_vals, abs_currents= zip(*GHad_results)

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(GHad_vals, abs_currents, marker='o')
# plt.xlabel("GHad (eV)")
# plt.ylabel("Max |Current Density| (mA/cm²)")
# plt.title("Max Current Density vs GHad")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# df = pd.DataFrame(GHad_results, columns=["GHad (eV)", "Max |Current| (mA/cm²)", "ThetaH Coverage", "ThetaStar Coverage"])
# print(df.to_string(index=False))

print(f"Max Current: {abs_currents[0]:.3f} mA/cm² at GHad = {GHad_vals[0]/(AvoNum * conversion_factor):.3f} eV")

