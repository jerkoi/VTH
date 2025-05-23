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
k_V = (10**2)*cmax
k_T = 10*cmax
beta = 0.5
GHad = F * -0.35 #free energy of hydrogen adsorption
partialPH2 = 1
UpperV = 0.60
LowerV = 0.1
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
    U_V = (-GHad/F) + (RT*np.log(thetaA_Star/thetaA_H))/F
    #U relies on the free energy of hydrogen adsorption plus the log of surface coverage (considered a concentration)
    return U_V

#reduction is FORWARD, oxidation is REVERSE, all variables are consistent with this
#rate_r0 is not the culprit, using the rate equation that works for Ram's code produces the same issues
def rates(t, theta):
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta #surface coverages again, acting as concentrations
    V = potential(t)  # Use t directly (scalar)
    U_V = eqpot(theta) #call function to find U for given theta

    ##Volmer Rate equation
    r_V = k_V * (thetaA_star ** (1 - beta)) * (thetaA_H ** beta) * np.exp(beta * GHad / RT) * (np.exp(-(beta) * F * (V - U_V) / RT) - np.exp((1 - beta) * F * (V - U_V) / RT))
    ##tafel rate equation
    r_T = k_T * (thetaA_H ** 2 * thetaA_star**2 * partialPH2 * np.exp((-2 * GHad) / RT))
    r_T = 0
    return r_V, r_T

def sitebal_r0(t, theta):
       r_V, r_T = rates(t, theta)
       dthetadt = [((2 * r_V) - r_T)  / cmax, (r_T - (2 * r_V)) / cmax] # [0 = star, 1 = H]
       return dthetadt

V = np.array([potential(ti) for ti in t])
curr1 = np.empty(len(t), dtype=object)
tcurr1= np.empty(len(t), dtype=object)

############################################################################################################################################################
############################################################################################################################################################
########################################################## SOLVER ##########################################################################################
############################################################################################################################################################
############################################################################################################################################################

soln = solve_ivp(sitebal_r0, duration, theta0, t_eval=t, method = 'LSODA')
print(soln.t.shape, soln.y.shape)

# Extract coverages from odeint
thetaA_Star = soln.y[0, :]
thetaA_H = soln.y[1, :]

#calculates rate based on theta values calculated during odeint, zips it with time given from potential(x) function
rate_vals = np.array([rates(time, theta) for time, theta in zip(t, soln.y.T)])
print("Rate size:", np.size(rate_vals))
curr1 = rate_vals * -F

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
plt.plot(t, thetaA_Star, label=r'$\theta_A^*$ (empty sites)', color='magenta')
plt.plot(t, thetaA_H, label=r'$\theta_A^H$ (adsorbed hydrogen)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Coverage')
plt.grid()
plt.legend()
plt.title('Surface Coverage vs. Time')
plt.show()



#Plot of reaction rate vs time
plt.figure(figsize=(8, 6))
plt.plot(t[1:], rate_vals[1:], label=r'$r_0$ (rate of hydrogen adsorption)', color='green')
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
    "R0": rate_vals,               # Reaction rate values
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


