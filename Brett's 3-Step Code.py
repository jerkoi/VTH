# Importing libraries- numpy, odeint, error function, matplotlib
import numpy as np
from scipy.integrate import odeint
from scipy.special import erf
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
# 0 will denote the Volmer step (r0, U0, dG0, etc)
# for thetas list: [empty sites, H sites]
# could have an issue with make dGk[2] value a negative?
# Gawas' code doesn't run the same every time

###############################################################################
# SETTING UP CONSTANTS, INITIAL CONDITIONS, ETC. #
##################################################

# physical constants and model parameters
RT = 8.314*298      # J/mol
F = 96485.0         # Faraday constant C/mol
cmax = 7.5e-10      # total available sites (mol. cm-2. s-1) (should be a very small number)
A = [1e5]           # list of pre-exponential terms relating to cmax
kf0 = A[0]*cmax     # kf0 is kf for hydrogen adsorption in acid
kf_list = [kf0, kf0, kf0 ]   # kf values for the Volmer, Heyrovsky, and Tafel steps respectively
beta = [0.5, 0.5, 0.5]

# need to give "odeint" a set of initial site coverages as a starting point to solve for successive site coverages
theta_H_0 = 0.1
theta_star_0 = 1.0 - theta_H_0
theta_0_list = [theta_star_0, theta_H_0]

# creates an array of time values to be used in time plots
t_stop = 100.0
t_inc = 0.05 # increment of time between measurements
t = np.arange(0.0, t_stop, t_inc)
ts = len(t)

# define diffusion limited current, calculated from levich equation for 1600 rpm O2, 0.196 cm2 Pt(111) disk in 0.1 M HClO4
# wut
iD = -6.12 # mA/cm2











#############################################################################################################
# PREPARING FUNCTIONS #
#######################

# Linear sweep voltammetrty- defining a potential as a function of time
def potential(x):
    UpperV = 0.40
    LowerV = 0.05
    scanrate = 0.03 #scan rate in V/s
    timescan=(UpperV-LowerV)/(scanrate)
    if x%(2*timescan)<timescan:
            Vapp = LowerV + scanrate*(x%((UpperV-LowerV)/(scanrate)))
    else:
            Vapp = UpperV - scanrate*(x%((UpperV-LowerV)/(scanrate)))
    return Vapp


# Defines the occupant-specific G of sites, NOT deltaG
def Gsites(theta_temp):
    theta_star, theta_H = theta_temp
    G_list = F * np.array([0, -0.35])  # adsorption free energy of [empty sites and Hads] (J/mol, not volts)
    G_list[1] += 25000*theta_H # accounts for Temkin isotherm (G changes due to surrounding adsorbed H interfering with bond strength)
    return (G_list)
'''
Gawas betrayed me, I can never trust again. Gads0 in his code are voltage values (eVs?), not G value (see by the fact that he multiplies them by F later)
'''


# Calculates G of adsorption for different site occupant changes(e.g. empty to H, or oH to O)
def Grxn(theta_temp):
    G_star, G_H = Gsites(theta_temp)    # calling Gsites to get the G for sites occupied by specific species (empty sites, H, etc.)
    dGk0 = G_H - G_star                 # difference in G bewteen H-occupied site and empty site   # called dGk to differentiate it from what I'll call "dG", which is dGk + dg(configuration)
    dGk_list = [dGk0]     # second and third values are negative because of how Grxn calculations shake out
    return(dGk_list)



# calculates equilibrium potential U for each type of reaction (dependant on coverage)
def EqPot(theta_temp):
    theta_star, theta_H = theta_temp
    dGk_list = Grxn(theta_temp)
    U0 = -dGk_list[0]/F + (RT*np.log(theta_star/theta_H)/F)             # Volmer (we divide dG by F just like in our derivation)
    U1 = -dGk_list[0]/F + (RT*np.log(theta_H/theta_star)/F)              # Heyrovsky
    U2 = -dGk_list[0]/F + (RT*np.log((theta_H**2)/(theta_star**2))/F)    # Tafel
    U_list = [U0,U1,U2]
    return(U_list)
'''
the equilibrium potential, U, IS NOT CONSTANT due to the change in dG due to the Temkin isotherm AND because of the change in configuration entropy because of site coverage
'''



r0_list=[]
r1_list=[]
r2_list=[]
# finds rates and stuff
# dGk is the k portion of dG = dGk + dGs
def rates(theta_temp,t):
    theta_star, theta_H = theta_temp
    
    V = potential(t)
    dGk_list = Grxn(theta_temp) # this dG is specifically calculated at equilibrium, since it is going into the exponential term tied to Kf, which is derived from the system at equilibrium
    U_list = EqPot(theta_temp)
    
    # these rates all use the same dGk value
    r0 = kf_list[0] * np.exp((beta[0]*dGk_list[0]/(RT))) * ((theta_star)**(1-beta[0])) * (theta_H**beta[0]) * (np.exp(((-1)*beta[0]*F*(V-U_list[0]))/(RT)) - np.exp(((1-beta[0])*F*(V-U_list[0])/(RT)))) # Volmer step
    r1 = kf_list[1] * np.exp((beta[1]*(dGk_list[0])/(RT))) * ((theta_H)**(1-beta[1])) * ((theta_star)**beta[1]) * (np.exp(((-1)*beta[1]*F*(V-U_list[1]))/(RT)) - np.exp(((1-beta[1])*F*(V-U_list[1])/(RT)))) # Heyrovsky step
    r2 = kf_list[2] * np.exp((beta[2]*(dGk_list[0])/(RT))) * ((theta_H)**(2*beta[2])) * ((theta_star)**(2-2*beta[2])) * (np.exp(((-1)*beta[2]*F*(V-U_list[2]))/(RT)) - np.exp(((1-beta[2])*F*(V-U_list[2])/(RT)))) # Tafel step
    
    r0_list.append(r0)
    r1_list.append(r1)
    r2_list.append(r2)
    print("r0 lsit: ",r0_list[1000:1010]); print()
    print("r1 list:",r1_list[1000:1010]); print()
    print("r2 list:",r2_list[1000:1010]); print()
    print()
    
    r_list = [r0, r1, r2]
    return(r_list)


# calculating dtheta/dt, which will be integrated to find coverage values
# funciton must be passed ttheta_temp becuase the site coverage values change with each odeint iteration, and those changes impact everything else
def site_balance(theta_temp, t, oprams):
    theta_star, theta_H = theta_temp
    r = rates(theta_temp, t)
    dtheta_dt = [(-r[0]+r[1]+r[2])/cmax, (r[0]-r[1]-r[2])/cmax] # [forward Volmer, reverse Volmer]
    return (dtheta_dt)


# suuuuuper important peice
theta_complete_list = odeint(site_balance, theta_0_list, t, args=(beta,))









############################################################################################################
# CREATING THE PLOTS #
######################

# setting up empty arrays for applied voltage, kinetic current density, and total current density for use in plots
V = np.empty(ts, dtype=object)
curr_kin = np.empty(ts, dtype=object)
curr_tot = np.empty(ts, dtype=object)
for i in range(ts):
    V[i] = potential(t[i])
    
    
# calculate current from our solution- this will be a function time (and also potential- since potential is a function of time)
curr_kin = list(range(ts))
curr_tot = list(range(ts))
for i in range(ts):
    theta_temp = theta_complete_list[i,:]
    Vapp = potential(t[i])
    R = rates(theta_temp, Vapp)
    curr_kin[i] = -1000*F*(R[0]+R[1]+R[2])
    curr_tot[i] = iD*curr_kin[i]/(iD+curr_kin[i]) #Koutecky-levich equation to get the total current density  


# plotting applied voltage vs. time
plt.plot(t[0:ts], V[0:ts], 'r')
plt.xlabel("Time (s)");   plt.ylabel("V vs.RHE  (J/C)")
plt.show()


# plotting coverage vs. time
plt.plot(t[10:ts], theta_complete_list[10:,0], 'g', label="*")
plt.plot(t[10:ts], theta_complete_list[10:,1], 'b', label="Hads")
plt.xlabel("Time (s)");   plt.ylabel("Coverage");   plt.legend(loc='best')
plt.show()


# plotting coverage as a function of applied voltage
plt.plot(V[10:], theta_complete_list[10:, 0], 'm', label='*')
plt.plot(V[10:], theta_complete_list[10:, 1], 'g', label='Hads')
plt.xlabel("V vs. RHE (V)");   plt.ylabel("Coverage");   plt.legend(loc="best")
plt.show()

# plotting total current density as a function of applid potential
plt.plot(V[10:ts], curr_tot[10:ts], "g")
plt.xlabel("E vsRHE (V)");   plt.ylabel("Total Current Density (mA/cm2)")
plt.show()

# plotting kinetic current density as a function of applied potential
plt.plot(V[10:ts], curr_kin[10:ts], "r")
plt.xlabel("V vs RHE (V)");   plt.ylabel("Kinetic Current Density (mA/cm2)")
plt.show()