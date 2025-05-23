import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'font.size': 14})

# Physical Constants
RT = 8.314*298 #ideal gas law times temperature
F = 96485.0 #Faraday constant, C/mol
cmax = 7.5*10e-10 #mol*cm-2*s-1

# Model Parameters
A = 1*10**2
k1 = A*cmax # forward rate constant
beta = 0.5
GHad = F * -0.35 #free energy of hydrogen adsorption
'''ONCE AGAIN I AM BETRAYED BY THE UNITS'''
UpperV = 0.60
LowerV = -0.1
scanrate = 0.025  #scan rate in V/s
timescan = (UpperV-LowerV)/(scanrate)
time_list = np.arange(0.0, 2*timescan, scanrate)
endtime = time_list[-1]
duration = [0, endtime]
'''endtime? is Alex a doomsday cultist? Keep at a distance physiscally and emotionally. Tell the cats to avoid for now.'''


# empty indexes
rate_index = []
time_index = [time_list]
U01_index = []
U02_index = []  
U0_index = []
U11_index = []
U12_index = []
U1_index = []
j0_index = []
exp11_index = []
exp21_index = []
j1_index = []
exp12_index = []
exp22_index = []
r0_index = []
r1_index = []

# extra lists for testing
all_times = []
theta_star_list = []
theta_H_list = []
all_thetas_list = []
print("Time size: ", np.size(time_index))


# Initial conditions
thetaA_H0 = 0.99  # Initial coverage of Hads, needs to be high as this is reduction forward
thetaA_Star0 = 1.0 - thetaA_H0  # Initial coverage of empty sites
theta0 = np.array([thetaA_Star0, thetaA_H0])


###############################################################################
###############################################################################
                                # FUNCTIONS #
###############################################################################
###############################################################################

#Linear sweep voltammetry- defining a potential as a function of time
def potential(x):
    if x%(2*timescan)<timescan:
            Vapp = LowerV + scanrate*(x%((UpperV-LowerV)/(scanrate)))
    else:
            Vapp = UpperV - scanrate*(x%((UpperV-LowerV)/(scanrate)))
    return Vapp


# Function to calculate U and Keq from theta, dG
def eqpot(theta):
    theta = np.asarray(theta)
    thetaA_Star, thetaA_H = theta # unpack surface coverage
    
    # Volmer
    U0_1 = (-GHad/F)
    U0_2 = (RT*np.log(thetaA_Star/thetaA_H))/F
    U0 = U0_1 + U0_2
    
    #Tafel
    U1_1 = (-GHad/(2*F))
    U1_2 = (RT*np.log(thetaA_Star**2/thetaA_H**2))/(2*F) #tafel
    U1 = U1_1 + U1_2
    
    # appending lists
    U0_index.append(U0)
    U01_index.append(U0_1)
    U02_index.append(U0_2)
    U1_index.append(U1)
    U11_index.append(U1_1)
    U12_index.append(U1_2)
    #U relies on the free energy of hydrogen adsorption plus the log of surface coverage (considered a concentration)
    return U0, U1


# reduction is FORWARD, oxidation is REVERSE, all variables are consistent with this
# r0 is volmer step, r1 is tafel step
def rates(t, theta):
    all_times.append(t)
    
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta #surface coverages again, acting as concentrations
    V = potential(t)  # Use t directly (scalar)
    
    # Volmer
    U0, U1 = eqpot(theta)
    j0 = k1 * (thetaA_star ** (1 - beta)) * (thetaA_H ** beta) * np.exp(beta * GHad / RT)
    exp1_1 = np.exp(-(beta) * F * (V - U0) / RT)
    exp2_1 = np.exp((1 - beta) * F * (V - U0) / RT)
    r0 =  j0 * (exp1_1 - exp2_1) #volmer rate
    
    # Tafel
    j1 = k1 * (thetaA_star ** (2*beta)) * (thetaA_H ** (2 - 2*beta)) * np.exp(beta * GHad / RT)
    exp1_2 = np.exp(((1-beta)*2*F*(V - U1)) / RT)
    exp2_2 = np.exp((-(beta)*2*F*(V - U1)) / RT)
    r1 = j1 * (exp1_2 - exp2_2) #tafel rate
    
    # appending lists
    j0_index.append(j0)
    exp11_index.append(exp1_1)
    exp21_index.append(exp2_1)
    j1_index.append(j1)
    exp12_index.append(exp1_2)
    exp22_index.append(exp2_2)
    r0_index.append(r0)
    r1_index.append(r1)
    return r0, r1 # returns Volmer and Tafel rates


def sitebal(t, theta):
    theta_star, theta_H = theta
    theta_star_list.append(theta_star)
    theta_H_list.append(theta_H)
    
    r0, r1 = rates(t, theta)     # Volmer rate, Tafel rate
    theta_star_rate = r1-r0      # summing all step rates based on how they affect theta_star
    theta_H_rate = r0-r1         # summing all step rates based on how they affect theta_H
    dthetadt = [theta_star_rate / cmax, theta_H_rate / cmax] # rate of change of empty sites and Hads
    return dthetadt
# whether a rate is positive or negative depends on the associated reaction affects each site balance
# Tafel (r1) is positive in theta_star_rate because as the T reaction moves forward, the number of empty sites increases
# Volmer (r0) is negative in theta_star_rate because as the V reaction moves forward, the number of empty sites decreases
# Tafel (r1) is negative in theta_H_rate because as the T reaction moves forward, the number of H-occupied sites decreases
# Volmer (r0) is positive in theta_H_rate because as the V reaction moves forward, the number of H-occupied sites increases


V = np.array([potential(ti) for ti in time_list])
curr1 = np.empty(len(time_list), dtype=object)
tcurr1= np.empty(len(time_list), dtype=object)


############################################################################################################################################################
############################################################################################################################################################
########################################################## SOLVER ##########################################################################################
############################################################################################################################################################
############################################################################################################################################################

soln = solve_ivp(sitebal, duration, theta0, t_eval = time_list, method = "RK45", full_output=True)

if not soln.success:
    print("Solver failed. Why are you so bad at this? Dingus. :", soln.message)
    exit()  # Stop further execution if ODE solver fails

# Extract coverages from solve_ivp
thetaA_Star = soln.y[0, :]
thetaA_H = soln.y[1, :]

#calculates rate based on theta values calculated during solve_ivp, zips it with time given from potential(x) function
rate_vals = np.array([rates(time, theta) for time, theta in zip(time_list, soln.y.T)])
total_rate_vals =    (rate_vals[:,0] + rate_vals[:,1]) * -F     # addes each corresponding Volmer and Tafel rate value to get a list of net rate values
'''I think this is where the issue was with the double plot on the kinetic current graph: since "rates" originally returned the net forward and net backward rates (forw. Volmer + forw. Tafel    &     back. Volmer + back Tafel), those two lists HAD to be negatives of each other'''

rate_vals_r0 = rate_vals[:, 0]
rate_vals_r1 = rate_vals[:, 1]

rate_forward = rate_vals_r1 - rate_vals_r0
rate_backward = rate_vals_r0 - rate_vals_r1

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
#Plot results


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
plt.plot(time_list, thetaA_Star[:len(time_list)], label=r'$\theta_A^*$ (empty sites)', color='magenta')
plt.plot(time_list, thetaA_H[:len(time_list)], label=r'$\theta_A^H$ (adsorbed hydrogen)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Coverage')
plt.grid()
plt.legend()
plt.title(f'Surface Coverage vs. Time, A = {A}')
plt.show()

#Plot of reaction rate vs time
plt.figure(figsize=(8, 6))
plt.plot(time_list[1:], rate_forward[1:], label=r'$r_0$ (rate of hydrogen adsorption)', color='green')
plt.plot(time_list[1:], rate_backward[1:], label=r'$r_1$ (rate of hydrogen desorption)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel(r'rate $ (mol/cm²/s)')
plt.legend()
plt.title(f'Reaction Rate vs. Time, A = {A}')
plt.grid()
plt.show()

# plot kinetic current density as a function of potential
plt.plot(V[10:], total_rate_vals[10:], 'b')
plt.xlabel('V vs RHE(V)')
plt.ylabel('Kinetic current (mA/cm2)')
plt.title(f'Kinetic Current vs Potential, A = {A}')
plt.grid()
plt.show()

# # rates vs V
# plt.figure(figsize=(8, 6))
# plt.plot(V[1:], rate_forward[1:], 'b')
# plt.plot(V[1:], rate_backward[1:], 'g')
# plt.show()


# Ensure thetaA_Star and thetaA_H are one-dimensional
thetaA_Star_flat = thetaA_Star.flatten()
thetaA_H_flat = thetaA_H.flatten()

# Create a dictionary to hold the data for the Excel file
data = {
    "Time (s)": time_list,
    # "Voltage (V)": V,
    # "U0 Volmer": U0_index[:len(t)],
    # "U0 Volmer Gad": U01_index[:len(t)],
    # "U0 Volmer Exp": U02_index[:len(t)],  # Include time as a reference
    # "U1 Tafel": U1_index[:len(t)],
    # "U11 Tafel Gad": U11_index[:len(t)],
    # "U12 Tafel Exp": U12_index[:len(t)],   # Include time as a reference                     # Reaction rate values for R0
    # "ThetaA_Star": thetaA_Star_flat[:len(t)],           # Surface coverage of empty sites
    # "ThetaA_H": thetaA_H_flat[:len(t)],                 # Same for exponential terms
    # "J0": j0_index[:len(t)],
    # "J1": j1_index[:len(t)],
    # "Exp1 1": exp11_index[:len(t)],
    # "Exp2 1": exp21_index[:len(t)],
    # "Exp1 2": exp12_index[:len(t)],
    # "Exp2 2": exp22_index[:len(t)],
    # "R0": r0_index[:len(t)],
    # "R1": r1_index[:len(t)],
    "Rate r0": rate_vals_r0,
    "Rate r1": rate_vals_r1,                   
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Export the DataFrame to an Excel file
df.to_excel("reaction_data.xlsx", index=False)

print("Data exported successfully to reaction_data.xlsx")
print('Time length:', len(time_list))
print('U0 index length:', len(U0_index))
print("Solution shape", np.size(soln))