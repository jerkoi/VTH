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
A = 1*10**-12
k1 = A*cmax
beta = 0.5
GHad = F * 0.9 #free energy of hydrogen adsorption
UpperV = -0.2
LowerV = -1.6
scanrate = 0.025 #scan rate in V/s
timestep = 0.01
timescan = (UpperV-LowerV)/(scanrate)
t = np.arange(0.0, 2*timescan, scanrate)
endtime = t[-1]
duration = [0, endtime]

#Empty indexes
rate_index = []
time_index = [t]
exp1_index = []
exp2_index = []
j0_index = []
U0_index = []
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
    U0 = (-GHad/F) + (RT*np.log(thetaA_Star/thetaA_H))/F 
    U0_index.append(U0)
    #U relies on the free energy of hydrogen adsorption plus the log of surface coverage (considered a concentration)
    return U0

#reduction is FORWARD, oxidation is REVERSE, all variables are consistent with this
def rates_r0(t, theta):
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta #surface coverages again, acting as concentrations
    V = potential(t)  # Use t directly (scalar)
    U0 = eqpot(theta) #call function to find U for given theta
    ##My Rate equations
    r0 = k1 * (thetaA_star ** (1 - beta)) * (thetaA_H ** beta) * np.exp(beta * GHad / RT) * (np.exp(-(beta) * F * (V - U0) / RT) - np.exp((1 - beta) * F * (V - U0) / RT))
    return r0

def sitebal_r0(t, theta):
       r0 = rates_r0(t, theta)
       dthetadt = [-r0 / cmax, r0 / cmax] # [0 = star, 1 = H]
       return dthetadt

V = np.array([potential(ti) for ti in t])
curr1 = np.empty(len(t), dtype=object)
tcurr1= np.empty(len(t), dtype=object)

############################################################################################################################################################
############################################################################################################################################################
########################################################## SOLVER ##########################################################################################
############################################################################################################################################################
############################################################################################################################################################

soln = solve_ivp(sitebal_r0, duration, theta0, t_eval=t)



#Plotting U0 as a function of time
U0_values = [eqpot(theta) for theta in soln.y.T]


# Extract coverages from odeint
thetaA_Star = soln.y[0, :]
thetaA_H = soln.y[1, :]

#calculates rate based on theta values calculated during odeint, zips it with time given from potential(x) function
r0_vals = np.array([rates_r0(time, theta) for time, theta in zip(t, soln.y.T)]) 
print("Rate size:", np.size(r0_vals))
curr1 = r0_vals * -F

# Find the indices of the maximum and minimum values for rate
max_curr_index = np.argmax(curr1)
min_curr_index = np.argmin(curr1)

# Find the corresponding times
time_max_curr = t[max_curr_index]
time_min_curr = t[min_curr_index]

# Calculate the voltages at these times
voltage_max_curr = potential(time_max_curr)
voltage_min_curr = potential(time_min_curr)

# Print the results
print(f"Voltage at max current: {voltage_max_curr}")
print(f"Voltage at min current: {voltage_min_curr}")

###########################################################################################################################
###########################################################################################################################
#Plot results
plt.figure(figsize=(8, 6))
plt.plot(V, thetaA_Star, label=r'$\theta_A^*$ (empty sites)', color='magenta')
plt.plot(V, thetaA_H, label=r'$\theta_A^H$ (adsorbed hydrogen)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Coverage')
plt.grid()
plt.legend()
plt.title(f'Surface Coverage vs. Time, A = {A}')
plt.show()

#plotting U0 and V vs time
plt.figure(figsize=(8, 6))
plt.plot(t, U0_values, label='Equilibrium Potential (V)', color='orange')
plt.plot(t, [potential(ti) for ti in t], label="Potential (V)", color = 'blue')
plt.xlabel('Time (s)')
plt.ylabel('Potential (V)')
plt.title('Potential vs. Time')
plt.grid()
plt.legend()
plt.show()


# #Plot of reaction rate vs time
# plt.figure(figsize=(8, 6))
# plt.plot(V[1:], r0_vals[1:], label=r'$r_0$ (rate of hydrogen adsorption)', color='green')
# plt.xlabel('Time (s)')
# plt.ylabel(r'$r_0$ (mol/cm²/s)')
# plt.legend()
# plt.title(f'Reaction Rate vs. Time, A = {A}')
# plt.grid()
# plt.show()

# plot kinetic current desnity as a function of potential
plt.plot(V[10:20000], curr1[10:20000], 'b')
plt.xlabel('V vs RHE(V)')
plt.ylabel('Kinetic current (mA/cm2)')
plt.title(f'Kinetic Current vs Potential, A = {A}')
plt.grid()
plt.show()

# #plot of exp1 and exp2 (first exponential term and second exponential term in rate eq) vs time
# plt.plot(t, exp1_index[:len(t)], label='Exp1')
# plt.plot(t, exp2_index[:len(t)], label = 'Exp2')
# plt.ylim(0.8,1.2)
# plt.xlim(1, 2)
# plt.ylabel('Exp Value')
# plt.xlabel('Time (s)')
# plt.grid()
# plt.legend()
# plt.title('Exp Terms vs Time')
# plt.show()


# #plot of exchange current density (J0) from rate eq vs time
# plt.plot(t, j0_index[:len(t)])
# plt.ylabel('Exchange Current Density')
# plt.xlabel('Time (s)')
# plt.title('Exchange Current Density vs Time')
# plt.show()

print('Exp1:', len(exp1_index))
print('Exp2:', len(exp2_index))
print('J0:', len(j0_index))
print('U0:',len(U0_index))

#Create a dictionary to hold the data for excel file 
data = {
    "Time (s)": t,
    "Voltage (V)": V,  
    "Eq Potential": U0_index[:len(t)],            # Include time as a reference
    "R0": r0_vals,               # Reaction rate values
    "J0": j0_index[:len(t)],     # Ensure the length matches the time array
    "Exp1": exp1_index[:len(t)],
    "Exp2": exp2_index[:len(t)],
    "ThetaA_Star": thetaA_Star,
    "ThetaA_H": thetaA_H,   # Same for exponential terms
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Export the DataFrame to an Excel file
df.to_excel("reaction_data.xlsx", index=False)

print("Data exported successfully to reaction_data.xlsx")


