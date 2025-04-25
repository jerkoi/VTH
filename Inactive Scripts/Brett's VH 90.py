'''might be missing negative in UF = deltaG. Try adjusting parameters first, but try switching beta coefficients after that'''


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'font.size': 14})


###########################################################################################################################
###########################################################################################################################
                                                    # PARAMETERS #
###########################################################################################################################
###########################################################################################################################
'''0 is for Volmer, 1 is for Heyrovsky'''

# Physical Constants
RT = 8.314*298 #ideal gas law times temperature
F = 96485.0 #Faraday constant, C/mol
cmax = 7.5*10e-10 #mol*cm-2*s-1

'''changing deltaG to be lower'''
# Model Parameters
A = [10, 100] # frequency factor (Volmer, Heyrovsky)
k = [A[0]*cmax, A[1]*cmax] # forward rate constants (Volmer, Tafel)
beta = [0.5, 0.5] # thingy
G_Hads = F * -0.1      # change in Gibbs free energy due to Hydrogen adsorption (this is the actual value itself: other negative signs are taken care of elsewhere)

# Initial conditions
theta_H0 = 0.99  # Initial coverage of Hads, needs to be high as this is reduction forward
theta_Star0 = 1.0 - theta_H0  # Initial coverage of empty sites
theta0 = np.array([theta_Star0, theta_H0])

# potent    ial sweep & time 
UpperV = 0.05
LowerV = -0.3
scanrate = 0.025  #scan rate in V/s
timescan = (UpperV-LowerV)/(scanrate)
max_time = 60
time_list = np.arange(0.0, max_time, scanrate)
endtime = time_list[-1]
duration = [0, endtime]
time_index = [time_list]
'''endtime? is Alex a doomsday cultist? Keep at a distance physiscally and emotionally. Tell the cats to avoid for now.'''


r0list = []
r1list = []








##########################################################################################################################
##########################################################################################################################
                                                    # FUNCTIONS #
##########################################################################################################################
##########################################################################################################################

def potential(x):
    if x % (2 * timescan) < timescan:
        return LowerV + scanrate * (x % timescan)  # Forward sweep (rising)
    else:
        return UpperV - scanrate * ((x - timescan) % timescan)  # Reverse sweep (falling)


# Function to calculate U and Keq from theta & dG(adsorption)
def eqpot(theta):
    theta = np.asarray(theta)
    theta_star, theta_H = theta
    
    # Volmer
    U_01 = -G_Hads/F
    U_02 = (RT/F) * np.log(theta_star/theta_H)
    U0 = U_01 + U_02
    
    # Heyrovsky
    U_11 = G_Hads/F
    U_12 = (RT/F) * np.log(theta_H/theta_star)
    U1 = U_11 + U_12
    return U0, U1


# reduction is FORWARD, oxidation is REVERSE, all variables are consistent with this
def rates(t, theta):
    theta = np.asarray(theta)
    theta_star, theta_H = theta #surface coverages again, acting as concentrations
    V = potential(t)
    U0, U1 = eqpot(theta)
    
    # Volmer
    j0 = k[0]  *  np.exp(beta[0]*G_Hads/RT)  *  theta_star**(1-beta[0])  *  (theta_H**beta[0])
    exp11 = np.exp(-beta[0] * F * (V-U0) / RT)
    exp12 = np.exp((1-beta[0]) * F * (V-U0) / RT)
    r0 = j0 * (exp11 - exp12) # full Volmer rate equation
    r0list.append(r0)
    
    # Heyrovsky
    j1 = k[1]  *  np.exp(-beta[1]*G_Hads/RT)  *  theta_star**beta[1]  *  theta_H**(1-beta[1])
    exp21 = np.exp(-beta[1] * F * (V-U1) / RT)
    exp22 = np.exp((1-beta[1]) * F * (V-U1) / RT)
    r1 = j1 * (exp21 - exp22) # full Heyrovsky rate equation
    r1list.append(r1)
    #print(r1list[-1])

    return r0, r1
    

'''ONLY DOING VOLMER FOR NOW'''
def sitebal(t, theta):
       r0, r1 = rates(t, theta)     # Volmer rate, Heyrovsky rate
       theta_star_rate = r1-r0      # summing all step rates based on how they affect theta_star
       theta_H_rate = r0-r1        # summing all step rates based on how they affect theta_H
       dthetadt = [theta_star_rate / cmax, theta_H_rate / cmax] # rate of change of empty sites and Hads
       return dthetadt
# whether a rate is positive or negative depends on the associated reaction affects each site balance
# Heyrovsky (r1) is positive in theta_star_rate because as the H reaction moves forward, the number of empty sites increases
# Volmer (r0) is negative in theta_star_rate because as the V reaction moves forward, the number of empty sites decreases
# Heyrovsky (r1) is negative in theta_H_rate because as the H reaction moves forward, the number of H-occupied sites decreases
# Volmer (r0) is positive in theta_H_rate because as the V reaction moves forward, the number of H-occupied sites increases


V = np.array([potential(ti) for ti in time_list])
k_curr = np.empty(len(time_list), dtype=object)


############################################################################################################################################################
############################################################################################################################################################
                                                    # SOLVER #
############################################################################################################################################################
############################################################################################################################################################

# solve_ivp
soln = solve_ivp(sitebal, duration, theta0, t_eval = time_list, method = "RK45", full_output=True)
# if not soln.success:
#     print("Solver failed. Why are you so bad at this? Dingus. :", soln.message)
#     exit()  # Stop further execution if ODE solver fails

# Extract coverages from solve_ivp
theta_star = soln.y[0, :]
theta_H = soln.y[1, :]

#calculates rate based on theta values calculated during solve_ivp, zips it with time given from potential(x) function
rate_vals = np.array([rates(time, theta) for time, theta in zip(time_list, soln.y.T)])
total_rate_vals =    (rate_vals[:,0] + rate_vals[:,1]) * -F     # adds each corresponding Volmer and Tafel rate value

rate_vals_r0 = rate_vals[:, 0]
rate_vals_r1 = rate_vals[:, 1]

rate_forward = rate_vals_r1 - rate_vals_r0
rate_backward = rate_vals_r0 - rate_vals_r1

k_curr = rate_vals[:, 0] * -F


###########################################################################################################################
###########################################################################################################################
                                                        # PLOTS #
###########################################################################################################################
###########################################################################################################################

# V vs time
plt.figure(figsize=(8, 6))
plt.plot(time_list, V, label='Voltage (V)', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid()
plt.legend()
plt.title(f'Voltage vs. Time, A = {A}')
plt.show()

# coverage vs time
plt.figure(figsize=(8, 6))
plt.plot(time_list, theta_star[:len(time_list)], label=r'$\theta_A^*$ (empty sites)', color='magenta')
plt.plot(time_list, theta_H[:len(time_list)], label=r'$\theta_A^H$ (adsorbed hydrogen)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Coverage')
plt.grid()
plt.legend()
plt.title(f'Surface Coverage vs. Time, A = {A}')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(V, k_curr[:len(time_list)], label='Current Density (A/cm²)', color='green')
plt.xlabel('Voltage (V)')
plt.ylabel('Current Density (A/cm²)')
plt.grid()
plt.legend()
plt.title(f'Current Density vs. Voltage, A = {A}')
plt.show()

# # coverage vs V
# plt.figure(figsize=(8, 6))
# plt.plot(V, theta_star[:len(time_list)], label=r'$\theta_A^*$ (empty sites)', color='magenta')
# plt.plot(V, theta_H[:len(time_list)], label=r'$\theta_A^H$ (adsorbed hydrogen)', color='blue')
# plt.xlabel('Voltage (V)')
# plt.ylabel('Coverage')
# plt.grid()
# plt.legend()
# plt.title(f'Surface Coverage vs. V, A = {A}')
# plt.show()



###########################################################################################################################
###########################################################################################################################
                                                # EXPORTING DATA #
###########################################################################################################################
###########################################################################################################################