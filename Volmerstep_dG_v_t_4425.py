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
A = 1*10**-7
k1 = A*cmax
beta = 0.5
UpperV = -1.5
LowerV = -0.2
scanrate = 0.025 #scan rate in V/s
timestep = 0.01
timescan = (UpperV-LowerV)/(scanrate)
t = np.arange(0.0, 2*timescan, scanrate)
endtime = t[-1]
duration = [0, endtime]
period = 2
dGmin = 0.65
dGmax = 0.9

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

def dGvt(t):
    '''varying deltaG between -0.2 and -0.15 every 10 seconds'''

    return dGmin if (t // period) % 2 == 0 else dGmax


#Function to calculate U and Keq from theta, dG
def eqpot(theta, GHad):
    theta = np.asarray(theta)
    thetaA_Star, thetaA_H = theta # unpack surface coverage
    U_V = (-GHad/F) + (RT*np.log(thetaA_Star/thetaA_H))/F 
    #U relies on the free energy of hydrogen adsorption plus the log of surface coverage (considered a concentration)
    return U_V

#reduction is FORWARD, oxidation is REVERSE, all variables are consistent with this
def rates_r0(t, theta):
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta #surface coverages again, acting as concentrations
    V = potential(t)  # Use t directly (scalar)
    GHad = F * dGvt(t)  # Get the current dG value based on time
    U0 = eqpot(theta, GHad) #call function to find U for given theta
    ##Volmer RDS, Heyrovsky / Tafel fast
    r_V = k1 * (thetaA_star ** (1 - beta)) * (thetaA_H ** beta) * np.exp(beta * GHad / RT) * (np.exp(-(beta) * F * (V - U0) / RT) - np.exp((1 - beta) * F * (V - U0) / RT))
    return r_V

def sitebal_r0(t, theta):
    r_V = rates_r0(t, theta)  # Get rates
    dthetadt = [-r_V / cmax, r_V / cmax] # [0 = star, 1 = H]
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
U0_values = [eqpot(theta, F * dGvt(ti)) for theta, ti in zip(soln.y.T, t)]


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
plt.plot(V[1:], thetaA_Star[1:], label=r'$\theta_A^*$ (empty sites)', color='magenta')
plt.plot(V[1:], thetaA_H[1:], label=r'$\theta_A^H$ (adsorbed hydrogen)', color='blue')
plt.xlabel('Voltage vs. RHE (V)')
plt.ylabel('Coverage')
plt.grid()
plt.legend()
plt.title(f'Surface Coverage vs. Voltage, Minimum and Max dG = {dGmin, dGmax}, Period = {period}')
plt.show()


#Plot of reaction rate vs time
plt.figure(figsize=(8, 6))
plt.plot(t[1:], thetaA_Star[1:], label=r'$\theta_A^*$ (empty sites)', color='magenta')
plt.plot(t[1:], thetaA_H[1:], label=r'$\theta_A^H$ (adsorbed hydrogen)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel(r'$r_0$ (mol/cm²/s)')
plt.legend()
plt.title(f'Coverage vs Time, Minimum and Max dG = {dGmin, dGmax}, Period = {period}')
plt.grid()
plt.show()

# plot kinetic current desnity as a function of potential
plt.plot(V[10:20000], curr1[10:20000], 'b')
plt.xlabel('V vs RHE(V)')
plt.ylabel('Kinetic current (mA/cm2)')
plt.title(f'Kinetic Current vs Potential, Minimum and Max dG = {dGmin, dGmax}, Period = {period}')
plt.grid()
plt.show()


# #Create a dictionary to hold the data for excel file 
# data = {
#     "Time (s)": t,
#     "Voltage (V)": V,  
#     "Eq Potential": U0_index[:len(t)],            # Include time as a reference
#     "R0": r0_vals,               # Reaction rate values
#     "J0": j0_index[:len(t)],     # Ensure the length matches the time array
#     "Exp1": exp1_index[:len(t)],
#     "Exp2": exp2_index[:len(t)],
#     "ThetaA_Star": thetaA_Star,
#     "ThetaA_H": thetaA_H,   # Same for exponential terms
# }

# # Convert the dictionary to a DataFrame
# df = pd.DataFrame(data)

# # Export the DataFrame to an Excel file
# df.to_excel("reaction_data.xlsx", index=False)

# print("Data exported successfully to reaction_data.xlsx")


