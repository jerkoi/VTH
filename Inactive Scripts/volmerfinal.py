import numpy as np
from scipy.integrate import odeint
from scipy.special import erf
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

# Physical Constants
RT = 8.314*298 #ideal gas law times temperature
F = 96485.0 #Faraday constant, C/mol
cmax = 1e-10 #mol*cm-2*s-1

# Model Parameters
A = 1e3
k1 = A*cmax
beta = 0.5
GHad = F * -0.5 #free energy of hydrogen adsorption

#Initial conditions
thetaA_H0 = 0.99  # Initial coverage of Hads, needs to be high as this is reduction forward
thetaA_Star0 = 1.0 - thetaA_H0  # Initial coverage of empty sites
theta0 = [thetaA_Star0, thetaA_H0]

#time bounds for integration
tStop = 60.0
tInc = 0.5
t = np.arange(0.0, tStop, tInc)
ts = int(tStop / tInc)

#Linear sweep voltammetry- defining a potential as a function of time
def potential(x):
    UpperV = 0.1
    LowerV = 0.7
    scanrate = 0.025 #scan rate in V/s
    timescan=(UpperV-LowerV)/(scanrate)
    if x%(2*timescan)<timescan:
            Vapp = LowerV + scanrate*(x%((UpperV-LowerV)/(scanrate)))
    else:
            Vapp = UpperV - scanrate*(x%((UpperV-LowerV)/(scanrate)))
    return Vapp

#Function to calculate U and Keq from theta, dG
#should delta G be negative? Are we using theta as a concentration term?
def eqpot(theta):
    thetaA_Star, thetaA_H = theta # unpack surface coverage
    U0 = (-GHad/F) + ((RT/F)*np.log((thetaA_Star)/thetaA_H))
    #U relies on the free energy of hydrogen adsorption plus the log of surface coverage (considered a concentration)
    return U0

#reduction is FORWARD, oxidation is REVERSE, all variables are consistent with this
#rate_r0 is not the culprit, using the rate equation that works for Ram's code produces the same issues
def rates_r0(theta, t):
    thetaA_star, thetaA_H = theta #surface coverages again, acting as concentrations
    V = potential(t)  # Use t directly (scalar)
    U0 = eqpot(theta)
    ex_cd = k1 * np.exp((beta * GHad) / RT) * ((thetaA_star) ** (1 - beta)) * (thetaA_H ** beta)
    exp = (np.exp(-beta * F * (V - U0) / RT) + np.exp((1 - beta) * F * (V - U0) / RT))
    r0 = ex_cd * exp
    print('r0 = ', r0)
    return r0

def sitebal_r0(theta, t):
       r0 = rates_r0(theta, t)
       dthetadt = [-r0 / cmax, r0 / cmax] # [0 = star, 1 = H]
       print('dthetadt = ', dthetadt)
       return dthetadt

V = np.array([potential(ti) for ti in t])
curr1 = np.empty(ts, dtype=object)
tcurr1= np.empty(ts, dtype=object) 

# Solve the ODE
soln = odeint(sitebal_r0, theta0, t, args=())

#Plotting U0 as a function of time
U0_values = [eqpot(theta) for theta in soln]

# Extract coverages
thetaA_Star = soln[:, 0]
thetaA_H = soln[:, 1]


# Calculate r0 over time
r0_vals = [rates_r0(theta, potential(time)) for theta, time in zip(soln, t)]

for i in range(ts):
    theta_temp=soln[i,:]
    Vapp = potential(t[i])
    R = rates_r0(theta_temp, Vapp)
    curr1[i]= F*R

# Calculate the derivative of r0 with respect to time using np.gradient
r0_derivative = np.gradient(r0_vals, t)

#Plot the derivative of r0 as a function of time
# plt.figure(figsize=(8, 6))
# plt.plot(t, r0_derivative, label=r"$\frac{d r_0}{dt}$", color='red')
# plt.xlabel('Time (s)')
# plt.ylabel(r'$\frac{d r_0}{dt}$ (mol/cm²/s²)')
# plt.title('Derivative of Reaction Rate vs. Time')
# plt.legend()
# plt.grid()
# plt.show()



# Plot results
plt.figure(figsize=(8,6))
plt.plot(V, r0_vals)
plt.show

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
plt.plot(t, r0, label=r'$r_0$ (rate of hydrogen adsorption)', color='green')
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
plt.show()