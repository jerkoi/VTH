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
A_V = 1*10**2
A_T = 1*10
partialpH2 = 1
k_V = A_V*cmax
k_T = A_T*cmax
beta = 0.5
GHad = F * -0.35 #free energy of hydrogen adsorption
UpperV = 0.60
LowerV = 0.1
scanrate = 0.025 #scan rate in V/s
timestep = 0.01
timescan = (UpperV-LowerV)/(scanrate)
t = np.arange(0.0, 2*timescan, scanrate)
endtime = t[-1]
duration = [0, endtime]

#Empty indexes
r_V_index = []
r_T_index = []
U_V_values = []
time_index = [t]
exp_index = []
i0_index = []
U0_index = []
print("Time size: ", np.size(time_index))

#Initial conditions
theta_max = 1
thetaA_H0 = 0.99  # Initial coverage of Hads, needs to be high as this is reduction forward
thetaA_Star0 = theta_max - thetaA_H0  # Initial coverage of empty sites
theta0 = np.array([thetaA_Star0, thetaA_H0])

#Linear sweep voltammetry- defining a potential as a function of time
def potential(x):
    if x%(2*timescan)<timescan:
            Vapp = LowerV + scanrate*(x%((UpperV-LowerV)/(scanrate)))
    else:
            Vapp = UpperV - scanrate*(x%((UpperV-LowerV)/(scanrate)))
    return Vapp

#Function to calculate U and Keq from theta, dG
def eqpot(theta):
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta # unpack surface coverage
    
    #volmer eq
    U_V = (-GHad/F) + (RT*np.log(thetaA_star/thetaA_H))/F
    U_V_values.append(U_V)
    return U_V
print('U_V Values:', U_V_values)

#reduction is FORWARD, oxidation is REVERSE, all variables are consistent with this
def rates(t, theta):
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta #surface coverages again, acting as concentrations
    V = potential(t)  # Use t directly (scalar)
    U_V = eqpot(theta) #call function to find U for given theta

    ##Volmer rate
    r_V = k_V * (thetaA_star ** (1 - beta)) * (thetaA_H ** beta) * np.exp(beta * GHad / RT) * (np.exp(-(beta) * F * (V - U_V) / RT) - np.exp((1 - beta) * F * (V - U_V) / RT))


    ##Tafel rate
    r_T = (k_T * (thetaA_H**2)) - (partialpH2 * (thetaA_star**2))

    r_V_index.append(r_V)
    r_T_index.append(r_T)
    return np.array([r_V, r_T]) 

print('R_V values:', r_V_index)
print('R_T values:', r_T_index)

def sitebal_r0(t, theta):
    r_V, r_T = rates(t, theta)
    dthetadt = [2*r_T - r_V / cmax, r_V - 2*r_T / cmax] # [0 = star, 1 = H]
    return dthetadt

V = np.array([potential(ti) for ti in t])
curr1 = np.empty(len(t), dtype=object)
tcurr1= np.empty(len(t), dtype=object)

############################################################################################################################################################
############################################################################################################################################################
########################################################## SOLVER ##########################################################################################
############################################################################################################################################################
############################################################################################################################################################

soln = solve_ivp(sitebal_r0, [0, endtime], theta0, t_eval=t, method='RK45', rtol=1e-8, atol=1e-10)


print("Theta values:\n", soln.y)
print("soln.t shape:", soln.t.shape)
print("soln.y shape:", soln.y.shape)

#Unpacking Volmer eq pot values
U_V_values = [eqpot([theta_star, theta_H]) for theta_star, theta_H in zip(soln.y[0], soln.y[1])]


# Extract coverages from odeint
thetaA_Star = soln.y[0, :]
thetaA_H = soln.y[1, :]

#calculates rate based on theta values calculated during odeint, zips it with time given from potential(x) function
r0_vals = np.array([rates(time, theta) for time, theta in zip(t, soln.y.T)]) 
print("Rate size:", np.size(r0_vals))
curr1 = r0_vals * -F

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
plt.figure(figsize=(8, 6))
plt.plot(t, thetaA_Star, label=r'$\theta_A^*$ (empty sites)', color='magenta')
plt.plot(t, thetaA_H, label=r'$\theta_A^H$ (adsorbed hydrogen)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Coverage')
plt.grid()
plt.legend()
plt.title('Surface Coverage vs. Time')
plt.show()

# #plotting U0 and V vs time
# plt.figure(figsize=(8, 6))
# plt.plot(t, U0_values, label='Equilibrium Potential (V)', color='orange')
# plt.plot(t, [potential(ti) for ti in t], label="Potential (V)", color = 'blue')
# plt.xlabel('Time (s)')
# plt.ylabel('Potential (V)')
# plt.title('Potential vs. Time')
# plt.grid()
# plt.legend()
# plt.show()


#Plot of reaction rate vs time
plt.figure(figsize=(8, 6))
plt.plot(t[1:], r0_vals[1:], label=r'$r_0$ (rate of hydrogen adsorption)', color='green')
plt.xlabel('Time (s)')
plt.ylabel(r'$r_0$ (mol/cm²/s)')
plt.legend()
plt.title('Reaction Rate vs. Time')
plt.grid()
plt.show()

# plot kinetic current desnity as a function of potential
plt.plot(V[10:20000], curr1[10:20000], 'b')
plt.xlabel('V vs RHE(V)')
plt.ylabel('Kinetic current (mA/cm2)')
plt.title('Kinetic Current vs Potential')
plt.grid()
plt.show()


# print('Exp1:', len(exp1_index))
# print('Exp2:', len(exp2_index))
# print('U0:',len(U0_index))

# #Create a dictionary to hold the data for excel file 
# data = {
#     "Time (s)": t,
#     "Voltage (V)": V,  
#     "Eq Potential Volmer": U_V_values[:len(t)],  # Equilibrium potential values
#     "Eq Potential Tafel": U_T_values[:len(t)],  # Equilibrium potential values
#     "Theta Star": thetaA_Star[:len(t)],  # Surface coverage of empty sites
#     "Theta H": thetaA_H[:len(t)],   
#     "Tafel Rate": r_T_index[:len(t)],
#     "Volmer Rate": r_V_index[:len(t)],          
# }

# # Convert the dictionary to a DataFrame
# df = pd.DataFrame(data)

# # Export the DataFrame to an Excel file
# df.to_excel("reaction_data.xlsx", index=False)

# print("Data exported successfully to reaction_data.xlsx")


