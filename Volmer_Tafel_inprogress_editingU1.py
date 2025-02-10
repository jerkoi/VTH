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
A = 1*10
k1 = A*cmax
beta = 0.025
GHad = F * -0.3 #free energy of hydrogen adsorption
UpperV = 0.60
LowerV = 0.1
scanrate = 0.025  #scan rate in V/s
timescan = (UpperV-LowerV)/(scanrate)
t = np.arange(0.0, 2*timescan, scanrate)
endtime = t[-1]
duration = [0, endtime]

#Empty indexes
rate_index = []
time_index = [t]
U0_index = []
U1_index = []
j0_index = []
exp11_index = []
exp21_index = []
j1_index = []
exp12_index = []
exp22_index = []
print("Time size: ", np.size(time_index))

#Initial conditions
thetaA_H0 = 0.99  # Initial coverage of Hads, needs to be high as this is reduction forward
thetaA_Star0 = 1.0 - thetaA_H0  # Initial coverage of empty sites
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
    thetaA_Star, thetaA_H = theta # unpack surface coverage
    #U0 = (-GHad/F) + (RT*np.log(thetaA_Star/thetaA_H))/F #volmer
    U1 = (-GHad/(2*F)) + (RT*np.log(thetaA_Star/thetaA_H))/(F) #tafel
    #U0_index.append(U0)
    U1_index.append(U1)
    #U relies on the free energy of hydrogen adsorption plus the log of surface coverage (considered a concentration)
    return U1

#reduction is FORWARD, oxidation is REVERSE, all variables are consistent with this
#r0 is volmer step, r1 is tafel step
def rates(t, theta):
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta #surface coverages again, acting as concentrations
    V = potential(t)  # Use t directly (scalar)
    U1 = eqpot(theta)
    # j0 = k1 * (thetaA_star ** (1 - beta)) * (thetaA_H ** beta) * np.exp(beta * GHad / RT)
    # exp1_1 = np.exp(-(beta) * F * (V - U0) / RT)
    # exp2_1 = np.exp((1 - beta) * F * (V - U0) / RT)
    # r0 =  j0 * (exp1_1 - exp2_1) #volmer rate
    j1 = k1 * (thetaA_star ** (2*beta)) * (thetaA_H ** (2 - 2*beta)) * np.exp(beta * GHad / RT)
    exp1_2 = np.exp(((1-beta)*2*F*(V - U1)) / RT)
    exp2_2 = np.exp(((beta)*2*F*(V - U1)) / RT)
    r1 = j1 * (exp1_2 - exp2_2) #tafel rate
    # j0_index.append(j0)
    # exp11_index.append(exp1_1)
    # exp21_index.append(exp2_1)
    j1_index.append(j1)
    exp12_index.append(exp1_2)
    exp22_index.append(exp2_2)
    return r1

def sitebal(t, theta):
       r1 = rates(t, theta)
       dthetadt = [(r1) / cmax, (r1) / cmax] # rate of change of empty sites and Hads
       return dthetadt

V = np.array([potential(ti) for ti in t])
curr1 = np.empty(len(t), dtype=object)
tcurr1= np.empty(len(t), dtype=object)

############################################################################################################################################################
############################################################################################################################################################
########################################################## SOLVER ##########################################################################################
############################################################################################################################################################
############################################################################################################################################################

soln = solve_ivp(sitebal, duration, theta0, t_eval = t, method = 'BDF')

if not soln.success:
    print("Solver failed:", soln.message)
    exit()  # Stop further execution if ODE solver fails

# Extract coverages from odeint
thetaA_Star = soln.y[0, :]
thetaA_H = soln.y[1, :]

#calculates rate based on theta values calculated during odeint, zips it with time given from potential(x) function
rate_vals = np.array([rates(time, theta) for time, theta in zip(t, soln.y.T)]) 
print("Rate length:", len(rate_vals))
curr1 = rate_vals * -F



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
plt.plot(t, V, label='Voltage (V)', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid()
plt.legend()
plt.title(f'Voltage vs. Time, A = {A}')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(t, thetaA_Star[:len(t)], label=r'$\theta_A^*$ (empty sites)', color='magenta')
plt.plot(t, thetaA_H[:len(t)], label=r'$\theta_A^H$ (adsorbed hydrogen)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Coverage')
plt.grid()
plt.legend()
plt.title(f'Surface Coverage vs. Time, A = {A}')
plt.show()

#Plot of reaction rate vs time
plt.figure(figsize=(8, 6))
plt.plot(t[1:], rate_vals[1:], label=r'$r_0$ (rate of hydrogen adsorption)', color='green')
plt.xlabel('Time (s)')
plt.ylabel(r'$r_0$ (mol/cm²/s)')
plt.legend()
plt.title(f'Reaction Rate vs. Time, A = {A}')
plt.grid()
plt.show()

# plot kinetic current desnity as a function of potential
plt.plot(V[10:20000], curr1[10:20000], 'b')
plt.xlabel('V vs RHE(V)')
plt.ylabel('Kinetic current (mA/cm2)')
plt.title(f'Kinetic Current vs Potential, A = {A}')
plt.grid()
plt.show()

# Ensure rate_vals is one-dimensional
rate_vals_flat = rate_vals.flatten()

# Ensure thetaA_Star and thetaA_H are one-dimensional
thetaA_Star_flat = thetaA_Star.flatten()
thetaA_H_flat = thetaA_H.flatten()

# Create a dictionary to hold the data for the Excel file
data = {
    "Time (s)": t,
    "Voltage (V)": V,
    #"Eq Potential Volmer": U0_index[:len(t)],  # Include time as a reference
    "Eq Potential Tafel": U1_index[:len(t)],   # Include time as a reference
    #"R0": rate_vals_flat[:len(t)],                      # Reaction rate values
    "ThetaA_Star": thetaA_Star_flat[:len(t)],           # Surface coverage of empty sites
    "ThetaA_H": thetaA_H_flat[:len(t)],                 # Same for exponential terms
    #"J0": j0_index[:len(t)],
    "J1": j1_index[:len(t)],
    #"Exp1 1": exp11_index[:len(t)],
    #"Exp2 1": exp21_index[:len(t)],
    "Exp1 2": exp12_index[:len(t)],
    "Exp2 2": exp22_index[:len(t)],                         
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Export the DataFrame to an Excel file
df.to_excel("reaction_data.xlsx", index=False)

print("Data exported successfully to reaction_data.xlsx")
print('Time length:', len(t))
print('U0 index length:', len(U0_index))
print('U1 index length:', len(U1_index))