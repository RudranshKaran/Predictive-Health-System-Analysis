import pandas as pd
import numpy as np
import os

# Define the target coordinates and maximum deviation
target_lat = 12.9767462
target_lon = 77.5659894
max_deviation = 0.01

# Path to the CSV file
file_path = os.path.join('Regional', 'data', 'regional_data.csv')

# Read the CSV file
df = pd.read_csv(file_path)

# Generate random deviations within the specified range
num_rows = len(df)
lat_deviations = np.random.uniform(-max_deviation, max_deviation, num_rows)
lon_deviations = np.random.uniform(-max_deviation, max_deviation, num_rows)

# Apply the deviations to create new coordinates around the target
df['latitude'] = target_lat + lat_deviations
df['longitude'] = target_lon + lon_deviations

# Round to 9 decimal places for consistency
df['latitude'] = df['latitude'].round(9)
df['longitude'] = df['longitude'].round(9)

# Save the updated CSV file
df.to_csv(file_path, index=False)

print(f"Updated {num_rows} coordinates in {file_path}")
print(f"New coordinates are scattered around ({target_lat}, {target_lon}) with max deviation of ±{max_deviation}")