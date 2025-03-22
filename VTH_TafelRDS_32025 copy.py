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
cmax = 7.5*10e-10 #mol*cm-2*s-1

# Model Parameters
k_V = cmax * 10**2
k_T = cmax * 10**0
k_H = cmax * 10**4
partialPH2 = 1
beta = 0.5
GHad = -0.20 * F

# potential sweep & time 
UpperV = 0.3
LowerV = -0.3
scanrate = 0.025  #scan rate in V/s
timescan = (UpperV-LowerV)/(scanrate)
max_time = 60
t = np.arange(0.0, max_time, scanrate)
endtime = t[-1]
duration = [0, endtime]
time_index = [t]

#Initial conditions
thetaA_H0 = 0.99  # Initial coverage of Hads, needs to be high as this is reduction forward
thetaA_Star0 = 1.0 - thetaA_H0  # Initial coverage of empty sites
theta0 = np.array([thetaA_Star0, thetaA_H0])

r_H_index = []

############################################################################################################################
############################################################################################################################
########################################################## FUNCTIONS #######################################################
############################################################################################################################
############################################################################################################################

# Ask user for mechanism choice
while True:
    mechanism_choice = input("Choose mechanism: Heyrovsky RDS (0), Tafel RDS (1), Volmer RDS (Heyrovsky) (2), or Volmer RDS (Tafel) (3)?")
    if mechanism_choice in ["1"]:
        break  # Exit the loop if input is valid
    print("Working on those! Please use 1 for now.")

# Convert to integer for logic checks
mechanism_choice = int(mechanism_choice)

#Linear sweep voltammetry- defining a potential as a function of time
def potential(x):
    if x%(2*timescan)<timescan:
            return LowerV + scanrate*((x - timescan) % timescan)
    else:
            return UpperV - scanrate*((x - timescan) % timescan)


#Function to calculate U and Keq from theta, dG
def eqpot(theta):
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta # unpack surface coverage
        
    ##Heyrovsky
    # U_11 = GHad/F
    # U_12 = (RT/F) * np.log(thetaA_H/thetaA_star)
    # U_H = U_11 + U_12

    ##Heyrovsky RDS Volmer Fast
    #U_H = GHad/F + (RT/F) * np.log(thetaA_H/thetaA_star)
    return
    

#reduction is FORWARD, oxidation is REVERSE, all variables are consistent with this
def rates_r0(t, theta):
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta #surface coverages again, acting as concentrations
    V = potential(t)  # Use t directly (scalar)
    U_H = eqpot(theta) #call function to find U for given theta
    

    ##Heyrovsky RDS, Volmer Fast
    #if mechanism_choice == 0:  
        # j1 = k_H  *  np.exp(-beta*GHad/RT)  *  thetaA_star**beta  *  thetaA_H**(1-beta)
        # exp21 = np.exp(-beta * F * (V-U_H) / RT)
        # exp22 = np.exp((1-beta) * F * (V-U_H) / RT)
        # r_H = j1 * (exp21 - exp22)
        
        #r_H = k_H * np.exp(((-beta) * GHad) / RT) * (thetaA_H**(1 - beta)) * (thetaA_star**beta) * (np.exp((((1 - beta) * F) / RT) * (V - U_H)) - np.exp(((-beta * F) / RT) * (V - U_H)))

    ##Tafel RDS, Volmer fast
    r_T = 0
    if mechanism_choice == 1:
        r_T = k_T * ((thetaA_H **2) - ((thetaA_star ** 2) * np.exp((-2*GHad) / RT)))
    
    ##Volmer RDS, Heyrovsky fast
    # r_V = 0
    # if mechanism_choice == 2:
    #     r_V = k_V * (thetaA_star ** (1 - beta)) * (thetaA_H ** beta) * np.exp(beta * GHad / RT) * (np.exp(-(beta) * F * (V - U_V) / RT) - np.exp((1 - beta) * F * (V - U_V) / RT))
    
    return r_T

def sitebal_r0(t, theta):
       r_T = rates_r0(t, theta)
    #    if mechanism_choice == 0:
    #         thetaStar_rate_H = r_H / cmax
    #         thetaH_rate_H = (-r_H) / cmax
    #         dthetadt = [(thetaStar_rate_H), thetaH_rate_H]
       if mechanism_choice == 1:
            theta_star_rate = r_T      # summing all step rates based on how they affect theta_star
            theta_H_rate = -r_T        # summing all step rates based on how they affect theta_H
            dthetadt = [theta_star_rate / cmax, theta_H_rate / cmax]
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

soln = solve_ivp(sitebal_r0, duration, theta0, t_eval = t, method = "RK45", full_output=True)

print(f"Solver success: {soln.success}")
if not soln.success:
    print(soln.message)
    exit()

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
print('r0_vals shape:', r0_vals.shape)
curr1 = r0_vals * (-F)

# volmer_rate = r0_vals[:, 0]
# tafel_rate = r0_vals[:, 1]

'''assuming that tafel has an effect on the overall rate.  I wasn't sure about this.  If not, rate should just be volmer step'''
# t_rate = volmer_rate - tafel_rate

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
#Plot results
plt.figure(figsize=(8, 6))
plt.plot(V[1:], thetaA_Star[1:], label=r'$\theta_A^*$ (empty sites)', color='magenta')
plt.plot(V[1:], thetaA_H[1:], label=r'$\theta_A^H$ (adsorbed hydrogen)', color='blue')
plt.xlabel('Voltage vs. RHE (V)')
plt.ylabel('Coverage')
plt.grid()
plt.legend()
plt.title('Surface Coverage vs. Time')
plt.show()


#Plot of reaction rate vs time
plt.figure(figsize=(8, 6))
plt.plot(t[1:], thetaA_Star[1:], label='Theta Star Coverage', color='red')
plt.plot(t[1:], thetaA_H[1:], label='Theta H Coverage', color='green')
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



#######################################################################################################################################################
################################################################# EXCEL EXPORT ################################################################################################
#######################################################################################################################################################

#Create a dictionary to hold the data for excel file 
data = {
    "Time (s)": t[:len(t)],
    "Voltage (V)": V[:len(t)],
    "Heyrovsky rate from Rate Equation": r_H_index[:len(t)],  
    "Heyrovsky Rate": r0_vals[:len(t)],
    "ThetaA_Star": thetaA_Star[:len(t)],
    "ThetaA_H": thetaA_H[:len(t)],
    "Current": curr1[:len(t)]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Export the DataFrame to an Excel file
df.to_excel("reaction_data.xlsx", index=False)

print("Data exported successfully to reaction_data.xlsx")


