#Importing libraries- numpy, odeint, error function, matplotlib
import numpy as np
from scipy.integrate import odeint
from scipy.special import erf
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

# Physical Constants
RT = 8.314*298 #ideal gas law times temperature
F = 96485 #Faraday constant, C/mol
cmax = 7.5e-10 #mol*cm-2*s-1

# Model Parameters
k1 = 1e5*cmax
kf = np.array([k1])
beta = np.array([0.5])
aH = 1

# Linear sweep voltammetry- defining a potential as a function of time
def potential(x):
    UpperV = 0.52
    LowerV = -0.1
    scanrate = 0.05 #scan rate in V/s
    timescan=(UpperV-LowerV)/(scanrate)
    if x%(2*timescan)<timescan:
            Vapp = LowerV + scanrate*(x%((UpperV-LowerV)/(scanrate)))
    else:
            Vapp = UpperV - scanrate*(x%((UpperV-LowerV)/(scanrate)))
    return Vapp

# Function to calculate U and Keq from theta, dG
def EqPot(theta_temp):
    theta_star, theta_H = theta_temp
    dGk_list = Grxn(theta_temp)
    U0 = -dGk_list[0]/F + (RT*np.log(theta_star/theta_H)/F) # we divide dG by F just like in our derivation
    U_list = [U0]
    return(U_list)


def Gsites(theta_temp):
    theta_star, theta_H = theta_temp
    G_list = F * np.array([0, -0.35])  # adsorption free energy of [empty sites and Hads] (J/mol, not volts)
    G_list[1] += 25000*theta_H # accounts for Temkin isotherm (G changes due to surrounding adsorbed H interfering with bond strength)
    return (G_list)

# def Gsites(theta_temp):
#     theta_star, theta_H = theta_temp
#     G_list = F * np.array([0, -0.35])  # adsorption free energy of [empty sites and Hads] (J/mol, not volts)
#     G_list[1] += 25000*theta_H # accounts for Temkin isotherm (G changes due to surrounding adsorbed H interfering with bond strength)
#     return (G_list)

# Function to calculate reaction free energies from adsorption free energies
def Grxn(theta_temp):
    G_star, G_H = Gsites(theta_temp) # calling Gsites to get the G for sites occupied by specific species (empty sites, H, etc.)
    dGk0 = G_H - G_star # difference in G bewteen H-occupied site and empty site   # called dGk to differentiate it from what I'll call "dG", which is dGk + dg(configuration)
    dGk_list = [dGk0]
    return(dGk_list)



# def rates_r0(theta, V):
#     thetaA_star, thetaA_H = theta
#     r0 = k1 * (1 - thetaA_H) * thetaA_H  # Very simple rate expression
#     return r0


def rates_r0(theta, V):
       thetaA_star, thetaA_H = theta
       dG0 = -RT * np.log(thetaA_H/thetaA_star)
       U0 = EqPot(theta_temp)
       #r0 = kf[0]**(1-beta[0])*k1**(beta[0])*(1-thetaA_H)**(1-beta[0])*(thetaA_H**beta[0])*(aH)**(1-beta[0])*(np.exp(((-beta[0]*F*(V-U0))/RT)-np.exp(((1-beta[0])*F*(V-U0))/RT)))
       r0 = kf[0] * np.exp((beta[0]*dG0/(RT))) * ((thetaA_star)**(1-beta[0])) * (thetaA_H**beta[0]) * (np.exp(((-1)*beta[0]*F*(V-U0))/(RT)) - np.exp(((1-beta[0])*F*(V-U0)/(RT)))) # Volmer step
       return r0

def sitebal_r0(theta, t, oprams):
       V = potential(t)
       r0 = rates_r0(theta, V)
       dthetadt = [-r0 / cmax, r0 / cmax]
       return dthetadt

#Initial conditions
thetaA_H0 = 0.1  # Initial coverage of Hads
thetaA_Star0 = 1.0 - thetaA_H0  # Initial coverage of empty sites
theta0 = [thetaA_Star0, thetaA_H0]

#time bounds for integration
tStop = 64.0
tInc = 0.01
t = np.arange(0.0, tStop, tInc)
ts = int(tStop / tInc)

E = np.array([potential(ti) for ti in t])
curr1 = np.empty(ts, dtype=object)
tcurr1= np.empty(ts, dtype=object) 

# Solve the ODE
soln = odeint(sitebal_r0, theta0, t, args=(beta,))

# Extract coverages
thetaA_Star = soln[:, 0]
thetaA_H = soln[:, 1]

# Calculate r0 over time
r0_vals = [rates_r0(theta, potential(time)) for theta, time in zip(soln, t)]

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(t, thetaA_Star, label=r'$\theta_A^*$ (empty sites)', color='magenta')
plt.plot(t, thetaA_H, label=r'$\theta_A^H$ (adsorbed hydrogen)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Coverage')
plt.legend()
plt.title('Surface Coverage vs. Time')
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(t, [potential(ti) for ti in t], label="Potential (V)")
plt.xlabel('Time (s)')
plt.ylabel('Potential (V)')
plt.title('Potential vs. Time')
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(t, r0_vals, label=r'$r_0$ (rate of hydrogen adsorption)', color='green')
plt.xlabel('Time (s)')
plt.ylabel(r'$r_0$ (mol/cm²/s)')
plt.legend()
plt.title('Reaction Rate vs. Time')
plt.grid()
plt.show()

for i in range(ts):
    theta_temp=soln[i,:]
    Vapp=potential(t[i])
    R=rates_r0(theta_temp, Vapp)
    curr1[i]=-1000*F*R

# plot kinetic current desnity as a function of potential
plt.plot(E[10:20000], curr1[10:20000], 'b')
plt.xlabel('E vs RHE(V)')
plt.ylabel('Kinetic current (mA/cm2)')
plt.show()
