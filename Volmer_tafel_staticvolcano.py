import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# import pandas as pd
plt.rcParams.update({'font.size': 14})

###########################################################################################################################
###########################################################################################################################
####################################################### PARAMETERS ########################################################
###########################################################################################################################
###########################################################################################################################

# Physical Constants
RT = 8.314*298 #ideal gas law times temperature
F = 96485.0 #Faraday constant, C/mol
cmax = 7.5*10e-10 #mol*cm-2*s-1

# Model Parameters
k_V = cmax * 10**-6
k_T = cmax * 10**-8
partialPH2 = 1
beta = 0.5
UpperV = 1
LowerV = 0
scanrate = 0.025 #scan rate in V/s
timestep = 0.01
timescan = 240 #seconds
t = np.arange(0.0, timescan, scanrate)
endtime = t[-1]
duration = [0, endtime]

GHad = -0.3 * F
period = 1000 #SECONDS
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
    if x%(2*timescan)<timescan:
            Vapp = LowerV + scanrate*(x%((UpperV-LowerV)/(scanrate)))
    else:
            Vapp = UpperV - scanrate*(x%((UpperV-LowerV)/(scanrate)))
    return -0.3

def dGvt(t):
    '''varying deltaG between -0.2 and -0.15 every 10 seconds'''

    return GHad
    #return dGmin if (t // period) % 2 == 0 else dGmax


#Function to calculate U and Keq from theta, dG
def eqpot(theta, GHad):
    theta = np.asarray(theta)
    thetaA_Star, thetaA_H = theta # unpack surface coverage
    U0 = (-GHad/F) + (RT*np.log(thetaA_Star/thetaA_H))/F 
    #U relies on the free energy of hydrogen adsorption plus the log of surface coverage (considered a concentration)
    return U0

#reduction is FORWARD, oxidation is REVERSE, all variables are consistent with this
def rates_r0(t, theta):
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta #surface coverages again, acting as concentrations
    V = potential(t)  # Use t directly (scalar)
    GHad = F * dGvt(t)  # Get the current dG value based on time
    U0 = eqpot(theta, GHad) #call function to find U for given theta

    ##Volmer Rate Equation
    r_V = k_V * (thetaA_star ** (1 - beta)) * (thetaA_H ** beta) * np.exp(beta * GHad / RT) * (np.exp(-(beta) * F * (V - U0) / RT) - np.exp((1 - beta) * F * (V - U0) / RT))
    
    ##Tafel Rate equation
    ##Tafel does not contribute to kinetic current, but does affect coverage of adsorbed hydrogen and free sites
    r_T = k_T * ((thetaA_H **2) - (partialPH2 * (thetaA_star ** 2) * np.exp((-2*GHad) / RT)))
    return r_V, r_T

def sitebal_r0(t, theta):
       r_V, r_T = rates_r0(t, theta)
       thetaStar_rate = ((-2*r_V) + r_T) / cmax
       thetaH_rate = ((2*r_V) - r_T) / cmax
       dthetadt = [(thetaStar_rate), thetaH_rate] # [0 = star, 1 = H]
       return dthetadt

V = np.array([potential(ti) for ti in t])
curr1 = np.empty(len(t), dtype=object)
tcurr1= np.empty(len(t), dtype=object)

############################################################################################################################################################
############################################################################################################################################################
########################################################## SOLVER ##########################################################################################
############################################################################################################################################################
############################################################################################################################################################
# List of GHad values to try
GHad_list = np.linspace(-1, 1, 40)
GHad_results = []

for GHad in GHad_list:
    print(f"Simulating for GHad = {GHad:.3f}")
    GHad_fixed = GHad  # update global for this simulation
    
    # Solve the system
    soln = solve_ivp(sitebal_r0, duration, theta0, t_eval=t, method='BDF')
    
    # Extract theta
    thetaA_Star = soln.y[0, :]
    thetaA_H = soln.y[1, :]

    # Recalculate rates
    r0_vals = np.array([rates_r0(time, theta) for time, theta in zip(t, soln.y.T)])
    curr1 = r0_vals[:, 0] * -F  # current from Volmer step

    max_current = np.max(np.abs(curr1))  # record absolute max current
    GHad_results.append((GHad, max_current))  # save result


############################################################################################################################################################
############################################################################################################################################################
########################################################## Value Extracting ################################################################################
############################################################################################################################################################
############################################################################################################################################################

# Extract coverages from odeint
thetaA_Star = soln.y[0, :]
thetaA_H = soln.y[1, :]

#calculates rate based on theta values calculated during odeint, zips it with time given from potential(x) function
r0_vals = np.array([rates_r0(time, theta) for time, theta in zip(t, soln.y.T)])
###takes only volmer rate to compute kinetic current density
curr1 = r0_vals[:, 0] * -F
maxcurrent = np.max(curr1) #finds max current density

volmer_rate = r0_vals[:, 0]
#tafel_rate = r0_vals[:, 1]

'''assuming that tafel has an effect on the overall rate.  I wasn't sure about this.  If not, rate should just be volmer step'''
#t_rate = volmer_rate

# # Find the indices of the maximum and minimum values for rate
# max_curr_index = np.argmax(curr1)
# min_curr_index = np.argmin(curr1)

# # Find the corresponding times
# time_max_curr = t[max_curr_index]
# time_min_curr = t[min_curr_index]

# # Calculate the voltages at these times
# voltage_max_curr = potential(time_max_curr)
# voltage_min_curr = potential(time_min_curr)

# # Print the results
# print(f"Voltage at max current: {voltage_max_curr}")
# print(f"Voltage at min current: {voltage_min_curr}")

###########################################################################################################################
###########################################################################################################################
########################################################## PLOTS ############################################################
###########################################################################################################################
###########################################################################################################################

# Unpack results
GHad_vals, abs_currents = zip(*GHad_results)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(GHad_vals, abs_currents, marker='o')
plt.xlabel("GHad (eV)")
plt.ylabel("Max |Current Density| (A/cm²)")
plt.title("Max Current Density vs GHad")
plt.grid(True)
plt.tight_layout()
plt.show()
