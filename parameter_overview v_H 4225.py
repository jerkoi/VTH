import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})


###########################################################################################################################
###########################################################################################################################
####################################################### PARAMETERS ########################################################
###########################################################################################################################
###########################################################################################################################

# Set global plot style
plt.rcParams.update({'font.size': 14})

###########################################################################################################################
####################################################### PARAMETERS ########################################################
###########################################################################################################################

# Constants
RT = 8.314 * 298  # ideal gas law times temperature (J/mol·K * K)
F = 96485.0       # Faraday constant, C/mol
cmax = 7.5e-9     # mol·cm^-2·s^-1

# Kinetic constants
k_V = cmax * 10**-12
k_T = cmax * 10**-2
k_H = cmax * 10**-14
beta = 0.5

# List of GHad values (in volts)
GHad_list = [0.65, 0.9]

# Surface coverage combinations
theta_values = [(0.01, 0.99), (0.5, 0.5), (0.99, 0.01)]

# Prepare data for export
all_data = []

# Loop over theta values
for thetaA_star, thetaA_H in theta_values:
    plt.figure(figsize=(8, 6))  # New plot for each theta group

    for GHad_V in GHad_list:
        GHad = GHad_V * F  # Convert to Joules/mol
        V = np.linspace(0, -2.5, num=22)

        # Compute equilibrium potentials
        U_V = (-GHad / F) + (RT * np.log(thetaA_star / thetaA_H)) / F
        U_H = (GHad / F) + (RT / F) * np.log(thetaA_H / thetaA_star)

        # Volmer rate
        r_V = k_V * (thetaA_star ** (1 - beta)) * (thetaA_H ** beta) * np.exp(beta * GHad / RT) * (
            np.exp(-beta * F * (V - U_V) / RT) - np.exp((1 - beta) * F * (V - U_V) / RT)
        )

        # Heyrovsky rate
        j1 = k_H * np.exp(-beta * GHad / RT) * thetaA_star ** beta * thetaA_H ** (1 - beta)
        r_H = j1 * (np.exp(-beta * F * (V - U_H) / RT) - np.exp((1 - beta) * F * (V - U_H) / RT))

        theta_star_rate = r_H - r_V
        theta_H_rate = r_V - r_H

        # Filtered values for plotting
        filtered_V = []
        filtered_r_theta_star = []
        filtered_r_theta_H = []

        for i in range(len(V)):
            # Include in plot only if both rates are <= 1000
            if abs(theta_star_rate[i]) <= 1000 and abs(theta_H_rate[i]) <= 1000:
                filtered_V.append(V[i])
                filtered_r_theta_star.append(theta_star_rate[i])
                filtered_r_theta_H.append(theta_H_rate[i])

            # Collect everything for Excel
            all_data.append({
                "Voltage (V)": V[i],
                "GHad (V)": GHad_V,
                "theta_star": thetaA_star,
                "theta_H": thetaA_H,
                "r_θ*": theta_star_rate[i],
                "r_θH": theta_H_rate[i]
            })

        # Plot filtered rates
        plt.plot(filtered_V, filtered_r_theta_star, label=rf"$\theta^*_{{rate}}, G_{{Had}}={GHad_V:.1f}V$", marker='o')
        plt.plot(filtered_V, filtered_r_theta_H, label=rf"$\theta_H{{rate}}, G_{{Had}}={GHad_V:.1f}V$", marker='x')

    # Format and show plot
    plt.title(f"θ* = {thetaA_star:.2f}, θH = {thetaA_H:.2f}")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Export to Excel
df = pd.DataFrame(all_data)
df.to_excel("HER_rates_export.xlsx", index=False)
print("✅ Data exported to 'HER_rates_export.xlsx'")



# rows = []
# for i in range(len(voltage_list)):
#     V_arr = voltage_list[i]
#     theta_star_arr = theta_star_list[i]
#     theta_H_arr = theta_H_list[i]
#     thetaA_star, thetaA_H = theta_values[i]
#     for j in range(len(V_arr)):
#         rows.append([
#             f"({thetaA_star:.2f}, {thetaA_H:.2f})",  # Theta label
#             V_arr[j],
#             theta_star_arr[j],
#             theta_H_arr[j]
#         ])

# # Print nicely
# headers = ["Theta Values", "Voltage (V)", "Theta Star Rate", "Theta H Rate"]
# print(tabulate(rows, headers=headers, floatfmt=".6e", tablefmt="grid"))
