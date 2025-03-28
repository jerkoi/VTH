# Brett's Volmer Code (H+ in acid)
# steal my code, I steal your cat   >:(
# Importing libraries- numpy, odeint, error function, matplotlib
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'font.size': 14})
# SUPER IMPORTANT EXPLAINATION OF THE CODE STUFF:     
# ODEINT takes the t values it is fed and modifies them slightly to test different theta solutions (so if you give it t=1, it might calculate using t=1.1, 1.01, 0.99, 1.04, etc.) until it finds a solution it likes. Every time it does this, it needs to re-calculate U and r, which is why (as seen in the terminal printout) the functions are called so many more times than seems necessary. The "full" lists contain all values calculated by the functions when the code is executed (a lot of them). One thing you can get ODEINT to output is the "nfe" (number of function evaluations) list: the total number of times ODEINT has iterated up to the point at which it reaches each "correct" answer. By using the values in the "nfe" list as indices, we can take the "full" lists of values we made (for U, r, j0, etc.) and extract only the values at those indices, which were calculated during an iteration that produced a "correct" answer.  Therefore, the "correct" lists contain ONLY the values calculated during what ODEINT considers a "correct" answer, and they are what is used for the plots.
# lots of debris in here for testing, but it can be removed pretty easily (good idea to just comment it out though)
# tested: all parameters remain unchanged throughout the iterations
# tested: new voltage sweep works, except the bounds may not be exact based on scanrate
# EqPot, rates, and site_balance are called the same number of times which is good
# check to make sure nothing is going wrong with slicing the V_list for the "correct" plots below
# maybe check out the "t = int(t)" line








###############################################################################
                # PARAMETERS, INITIAL CONDITIONS, LISTS #
###############################################################################


# lists used in calculations
V_list = []             # voltages for potential sweep
time_list = []          # list of time stamps
curr_kin_list = []      # kinetic current values


# lists used for code testing
j0_full_list = []       # exchange current density
exp_1_full_list = []    # first exponential term in rate equation     
exp_2_full_list = []    # second exponential term in rate equation
exp_terms_full_list = []# full exponential term of rate equation (also shows the difference between the exponential terms)
U_full_list = []        # full lists of eq. potentials
r_full_list = []        # full list of rate values
dtheta_dt_list = []
time_list_extras = []   # list of ALL times ODEINT calls the functions for, NOT just the time values it is fed
EqPot_count = []        # number of times EqPot is called
rates_count = []        # number of times rates is called
site_bal_count = []     # number of times site_balance is called


# lists of "correct" values (also used for testing)
time_correct_list = []  # ODEINT runs the functions more times than there are time_list values: this is a list of the time values at which ODEINT got the "correct" answer for each step
U_correct_list = []         # correct eq. potential terms
j0_correct_list = []        # correct current exchange density terms
exp_1_correct_list = []     # correct first exponential terms in rate equation
exp_2_correct_list= []      # correct second exponential terms in rate equation
exp_terms_correct_list = [] # correct exponential term in rate equation (also shows the difference between the exponential terms)
r_correct_list = []         # correct rate equation values
overpotential_list = []     # overpotentials


# physical constants and model parameters
RT = 8.314*298      # J/mol
F = 96485.0         # Faraday constant (C/mol)
cmax = 7.5e-10      # total available sites per area (mol. cm-2. s-1)
A = 1e5             # pre-exponential term tied to cmax
kf = A*cmax         # forward rate constant (units?)
beta = 0.5          # symmetry factor
G_Hads = F * -0.5   # ChatGPT says about 0.2eV, butthis says 2.47eV? ADSORPTION OF HYDROGEN ON A Pt (111) SURFACE, K. CHRISTMANN, G. ERTL and T. PIGNET, Institut ftir Physikalische Chemie der Universitiit, Miinchen, W. Germany 


# initial site coverages (need to give "ODEINT" a set of initial site coverages as a starting point to solve for successive site coverages)
theta_H_0 = 0.99                            # H+ coverage (since we're sweeping from low to high voltage, we choose H+ coverage to start high)
theta_star_0 = 1.0 - theta_H_0              # empty site coverage
theta_0_list = [theta_star_0, theta_H_0]


# voltage sweep
# creates lists of V and t values
UpperV = 0.8       # upper sweep limit (eV)
LowerV = 0.2        # lower sweep limit (eV)
StartV = UpperV     # determines initial direction of sweep
cycles = 3          # how many full cycles the sweep will go through
scanrate = 0.005    # scanrate needs to be low enough for the probelm to actually be solved (lower values = more iterations for solving?)
points = int(((UpperV-LowerV)/scanrate)*cycles*2)+1 # the 2 is for going back and forth
switch = 0          # 0 means "voltage will go down", 1 means "voltage will go up" (can be either to start)
V_list.append(StartV)
for i in range(0,points):
    time_list.append(float(i))
    if StartV >= UpperV:
        switch = 0
    if StartV <= LowerV:
        switch = 1
    if switch == 0:
        StartV -= scanrate
    if switch == 1:
        StartV += scanrate
    V_list.append(StartV)
time_list.append(points)







###############################################################################
                            # FUNCTIONS #
###############################################################################


# calculates equilibrium potential (U) for each iteration (dependant on coverage)
# the equilibrium potential, U, IS NOT CONSTANT due to the change in applied potential
def EqPot(theta_list):
    EqPot_count.append(1) # counting times function is called
    theta_star, theta_H = theta_list
    U = -G_Hads/F + (RT*np.log(theta_star/theta_H))/F # we divide dG by F just like in our derivation
    return(U)


# calculating the rate equation value
def rates(theta_list,t):
    rates_count.append(1)           # counting times function is called
    theta_star,theta_H = theta_list
    U = EqPot (theta_list)
    t = int(t)                      # converts t from float to int for use as an index (change this? seems unhealthy)
    V = V_list[t]                   # applied potential


    # making the rate equation
    j0 = kf * np.exp((beta*G_Hads)/RT) * ((theta_star)**(1-beta)) * (theta_H**beta)     # exchange current density
    first_exp_term = np.exp(-beta*F*(V-U)/RT)       # first exponential term
    second_exp_term = np.exp((1-beta)*F*(V-U)/RT)   # second exponential term                
    exp_terms = first_exp_term - second_exp_term    # full exponential term
    r = -j0*(F/RT)*(V-U)#*exp_terms                 # full rate equation for Volmer step
    
    # appending lists for testing
    j0_full_list.append(j0)
    exp_1_full_list.append(np.exp(-beta*F*(V-U)/RT))
    exp_2_full_list.append(-np.exp((1-beta)*F*(V-U)/RT))
    exp_terms_full_list.append(exp_terms)
    r_full_list.append(r)
    U_full_list.append(U)

    return(r)


# calculating dtheta/dt, which will be integrated to find coverage values
# funciton must be passed theta_list becuase the site coverage values change with each odeint iteration, and those changes impact everything else
def site_balance(theta_list, t, oprams):
    site_bal_count.append(1)            # counting times function is called
    time_list_extras.append(t)
    theta_star, theta_H = theta_list
    r = rates(theta_list, t)
    dtheta_dt = [-r/cmax, r/cmax]       # [forward Volmer, reverse Volmer]
    dtheta_dt_list.append(dtheta_dt)
    return (dtheta_dt)









    
###############################################################################
                            # CALCULATIONS #  
###############################################################################

# solves the ODE created by the rate eqution and site balance stuff
theta_complete_list, info = odeint(site_balance, theta_0_list, time_list, args=(beta,), full_output=True)


# Convert the `info` dictionary to a pandas DataFrame
# Export to an Excel file
info_df = pd.DataFrame({key: [value] for key, value in info.items()})
info_df.to_excel("info_output3.xlsx", index=False)


# creating lists of "correct" values
nfe_list = info['nfe']  # "number of function evaluations": the total number of times ODEINT has run up to the point at which it reaches each "correct" answer
for i in nfe_list:
    f = time_list_extras[i-1]
    time_correct_list.append(time_list_extras[i-1])
    U_correct_list.append(U_full_list[i-1])
    j0_correct_list.append(j0_full_list[i-1])
    exp_1_correct_list.append(exp_1_full_list[i-1])     # first exponential term in rate equation     
    exp_2_correct_list.append(exp_2_full_list[i-1])     # second exponential term in rate equation
    exp_terms_correct_list.append(exp_terms_full_list[i-1])     # full exponential term of rate equation (also shows the difference between the exponential terms)
    r_correct_list.append(r_full_list[i-1])


# calculating kinetic current values
for i in range(0, len(time_list)-1):
    r = r_correct_list[i]
    curr_kin = -1000*F*r            # factor of 1000 is to convert Amps to mA
    curr_kin_list.append(curr_kin)







###############################################################################
                                # PLOTS #
###############################################################################

# applied voltage vs. time
plt.plot(time_list[0:], V_list[0:], 'r')
plt.xlabel("Time (s)")
plt.ylabel("V vs.RHE  (J/C)")
plt.title("Applied Voltage vs. Time")
plt.show()

# coverage vs. time
plt.plot(time_list[0:], theta_complete_list[0:,0], 'g', label="*")
plt.plot(time_list[0:], theta_complete_list[0:,1], 'b', label="Hads")
plt.xlabel("Time (s)")
plt.ylabel("Coverage")
plt.legend(loc='best')
plt.title('Coverage vs Time')
plt.show()

# coverage as a function of applied voltage
plt.plot(V_list[0:], theta_complete_list[0:, 0], 'm', label='*')
plt.plot(V_list[0:], theta_complete_list[0:, 1], 'g', label='Hads')
plt.xlabel("V vs.RHE  (J/C)")
plt.ylabel("Coverage")
plt.legend(loc="best")
plt.title('Coverage vs. Applied Voltage')
plt.show()










###############################################################################
                        # CODE TESTING #
###############################################################################


print()
print("time_list length:", len(time_list))
print('V_list length:', len(V_list))
print("r_full_list length:", len(r_full_list))
print("U_full_list length:", len(U_full_list))
print("j0_full_list length:", len(j0_full_list))
print("exp_terms_full_list length:", len(exp_terms_full_list))
print("first exp full list length:", len(exp_1_full_list))
print('second exp full list length:', len(exp_2_full_list))
print('overpotential_list length:', len(overpotential_list))
print()
print('times EqPot is called:', len(EqPot_count))
print('times rates is called:', len(rates_count))
print('times site_balance is called:', len(site_bal_count))
print()
print("r_correct_list length:", len(r_correct_list))
print("j0_correct_list length:", len(j0_correct_list))
print("exp_terms_correct_list length:", len(exp_terms_correct_list))
print("first exp correct list length:", len(exp_1_correct_list))
print('second exp correct list length:', len(exp_2_correct_list))
print('overpotential_list length:', len(overpotential_list))
print()


# rate vs voltage
plt.plot(V_list[1:], r_correct_list, 'g')
plt.xlabel('V vs.RHE  (J/C)')
plt.ylabel('Rate (M/s)')
plt.title('Correct Rate vs V')
plt.show()

# exchange current density (j0) vs voltage
plt.plot(V_list[1:], j0_correct_list, 'b')  # since j0_list is appended through multiple function calls, we need to use the right section of it
plt.xlabel("V vs.RHE  (J/C)")
plt.ylabel("j0 (mA/m2)")
plt.title("Correct Exchange Current Density vs V")
plt.show()

# full exponential rate term vs voltage
plt.plot(V_list[101:], exp_terms_correct_list[100:], 'r')
plt.xlabel('V vs.RHE  (J/C)')
plt.ylabel('exp term')
plt.title('Correct Full Exponential term vs V')
plt.show()

# first exponential term vs voltage
plt.plot(V_list[1:], exp_1_correct_list, 'g')
plt.xlabel('V vs.RHE  (J/C)')
plt.ylabel('first exp term')
plt.title('Correct First Exp Term vs V')
plt.show()

# second exponential term vs voltage
plt.plot(V_list[1:], exp_2_correct_list, 'p')
plt.xlabel('V vs.RHE  (J/C)')
plt.ylabel('second exp term')
plt.title('Correct Second Exp Term vs V')
plt.show()

# equillibrium potential vs votlage
plt.plot(V_list[1:], U_correct_list, 'm')
plt.xlabel('V vs.RHE  (J/C)')
plt.ylabel('U (Volts)')
plt.title('Correct Equilibrium Potential vs V')
plt.show()

# equillibrium potential vs time
plt.plot(time_correct_list, U_correct_list, 'v')
plt.xlabel('Time (s)')
plt.ylabel('U (Volts)')
plt.title('Correct Equilibrium Potential vs Time')
plt.show()
    
plt.plot(time_list[1:], curr_kin_list, 'r')
plt.xlabel('time')
plt.ylabel('kinetic current (mA/m2)')
plt.title('Kinetic Current vs Time')
plt.show()







###############################################################################
                            # TESTING NOTES #
###############################################################################

# checked "PARAMETERS' section, everything stays the same throughout the code
# 'rate vs V' and 'kcurr vs V' plots are the same form, just scaled differently which is correct. But they're not the right values
# something might be wrong with the 'potential' function

# rate never goes negative, which it should
    # j0 never goes negative and is smooth, which is good
    # full exponential term never goes negative, which is bad because it should
    # second exponential term is very small in scale compared to the the first, which means the rate function is always dominated by the positive term (bad)

# questions for meeting:
    # a ewe sure -0.35 is the correct free energy of adsoroption?
    # no calculatations in plotting area
    # all initializing at top
    # should t be based on V? can we flip things to feed V to ODEINT instead of T?
# the index "-1" denotes "the last index in a list"



    