#Importing libraries- numpy, odeint, error function, matplotlib
#check units?
import numpy as np
from scipy.integrate import odeint
from scipy.special import erf
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

# Physical Constants
RT = 8.314*298 #ideal gas law times temperature
F = 96485.0 #Faraday constant, C/mol
cmax = 7.5e-10 #mol*cm-2*s-1

# Model Parameters
A = [1e2]
k1 = A[0]*cmax
kf = np.array([k1])
beta = np.array([0.5, 0.5, 0.5])
aH = 1
GHad = F * -0.35


#Initial conditions
thetaA_H0 = 0.99  # Initial coverage of Hads
thetaA_Star0 = 1.0 - thetaA_H0  # Initial coverage of empty sites
theta0 = [thetaA_Star0, thetaA_H0]

#time bounds for integration
tStop = 60.0
tInc = 0.01
t = np.arange(0.0, tStop, tInc)
ts = int(tStop / tInc)

#Linear sweep voltammetry- defining a potential as a function of time
def potential(x):
    UpperV = 0.52
    LowerV = -0.1
    scanrate = 0.025 #scan rate in V/s
    timescan=(UpperV-LowerV)/(scanrate)
    if x%(2*timescan)<timescan:
            Vapp = LowerV + scanrate*(x%((UpperV-LowerV)/(scanrate)))
    else:
            Vapp = UpperV - scanrate*(x%((UpperV-LowerV)/(scanrate)))
    return Vapp

# Linear sweep voltammetry- defining a potential as a function of time
# def potential(x):
#     UpperV = 0.52
#     LowerV = -0.1
#     scanrate = 0.05  # scan rate in V/s
#     timescan = (UpperV - LowerV) / scanrate
#     if x % (2 * timescan) < timescan:
#         Vapp = LowerV + scanrate * (x % timescan)
#     else:
#         Vapp = UpperV - scanrate * (x % timescan)
#     return Vapp


#G adsorption of H at empty sites
# def Gad(theta):
#     Gads = F * np.array([0, -0.35]) #adsorption free energies for [free sitesA, Had , OHad, Oad] #change
#     #Gads[1] += 25000*thetaA_H #linear tempkin isotherm
#     return Gads

# Function to calculate reaction free energies from adsorption free energies
# def Grxn(Gads):
#     GadA_Star, GadA_H = Gad(Gads)
#     dGr0 = GadA_H - GadA_Star
#     #dGr0 = Gads[1] - Gads[0]
#     return dGr0

# Function to calculate U and Keq from theta, dG
#should delta G be negative? Are we using theta as a concentration term?
def eqpot(theta):
    thetaA_Star, thetaA_H = theta # unpack surface coverage
    U0 = (-GHad/F) + (RT*np.log(thetaA_Star/thetaA_H))/F
    return U0

#reduction is FORWARD, oxidation is REVERSE, all variables are consistent with this
def rates_r0(theta, t):
    thetaA_star, thetaA_H = theta
    V = potential(t)  # Use t directly (scalar)
    U0 = eqpot(theta)
    ex_cd = kf[0] * np.exp((beta[0] * GHad) / RT) * ((thetaA_star) ** (1 - beta[0])) * (thetaA_H ** beta[0])
    exp = (np.exp((-1) * beta[0] * F * (V - U0) / RT) - np.exp((1 - beta[0]) * F * (V - U0) / RT))
    r0 = ex_cd * exp
    return r0


def sitebal_r0(theta, t, oprams):
       r0 = rates_r0(theta, t)
       dthetadt = [-r0 / cmax, r0 / cmax] # [0 = star, 1 = H]
       return dthetadt

E = np.array([potential(ti) for ti in t]) #E and Vapp need to be consolidated
curr1 = np.empty(ts, dtype=object)
tcurr1= np.empty(ts, dtype=object) 

# Solve the ODE
soln = odeint(sitebal_r0, theta0, t, args=(beta,))

# Extract coverages
thetaA_Star = soln[:, 0]
thetaA_H = soln[:, 1]

# Calculate r0 over time
r0_vals = [rates_r0(theta, potential(time)) for theta, time in zip(soln, t)]

for i in range(ts):
    theta_temp=soln[i,:]
    Vapp=potential(t[i])
    R=rates_r0(theta_temp, Vapp)
    curr1[i]=-1000*F*R

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

plt.plot(E[10:], thetaA_Star[10:], 'm', label='*')
plt.plot(E[10:], thetaA_H[10:], 'g', label='Hads')
plt.xlabel("V vs. RHE (V)")
plt.ylabel("Coverage")
plt.legend(loc="best")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(t, r0_vals, label=r'$r_0$ (rate of hydrogen adsorption)', color='green')
plt.xlabel('Time (s)')
plt.ylabel(r'$r_0$ (mol/cm²/s)')
plt.legend()
plt.title('Reaction Rate vs. Time')
plt.grid()
plt.show()


# plot kinetic current desnity as a function of potential
plt.plot(E[10:20000], curr1[10:20000], 'b')
plt.xlabel('E vs RHE(V)')
plt.ylabel('Kinetic current (mA/cm2)')
plt.show()
plt.plot(t, curr1, 'g')
plt.show()

# Define ex_cd array
ex_cd = np.empty(ts, dtype=object)

# Calculate ex_cd from the rate equation
for i in range(ts):
    theta_temp = soln[i, :]
    Vapp = potential(t[i])  # Calculate the potential at time step i
    r0 = rates_r0(theta_temp, Vapp)  # Calculate the reaction rate (r0)
    
    # Normalize or define ex_cd from r0 directly
    ex_cd[i] = r0 / np.max(r0)  # Normalize ex_cd by maximum reaction rate (or use another formula)

# Plot ex_cd vs Voltage (E)
plt.figure(figsize=(8, 6))
plt.plot(E[10:], ex_cd[10:], 'b', label='ex_cd (Reaction Rate Normalized)')
plt.xlabel('Potential (V)')
plt.ylabel('ex_cd (Normalized Reaction Rate)')
plt.title('ex_cd vs Voltage')
plt.legend()
plt.grid()
plt.show()
