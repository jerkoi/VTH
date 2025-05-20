import numpy as np

def generate_t_and_range(period, base_length=1000, base_period=0.5):
    # Inversely scale the length of t with respect to period
    scale_factor = base_period / period
    length = int(base_length * scale_factor)

    # Create the time array t
    t = np.linspace(0, 20, length)

    # Example range_array: binary toggle based on period
    range_array = ((t % period) < (period / 2)).astype(int)

    return t, range_array

# Example: Smaller period → more data points
t_small, r_small = generate_t_and_range(0.25)

# Example: Larger period → fewer data points
t_large, r_large = generate_t_and_range(1.0)

print(f"Small period (0.25): t length = {len(t_small)}, r length = {len(r_small)}")
print(f"Large period (1.0): t length = {len(t_large)}, r length = {len(r_large)}")