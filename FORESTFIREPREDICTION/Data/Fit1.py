import pandas as pd
import numpy as np

# 1. Load your newly created 8-year master dataset
df = pd.read_csv('ForestFire_8_Year_Master_Dataset.csv')

print("Original Static Data (Notice how it repeats perfectly):")
print(df[['Date', 'Elevation_m', 'Slope_deg', 'Baseline_NDVI']].head(3))

# Set a random seed so the "randomness" is exactly the same every time you run it
np.random.seed(42)

# 2. Add realistic daily variance (Gaussian Noise)

# Elevation: +/- ~1.5 meters of "GPS/Barometric error"
df['Elevation_m'] = np.round(df['Elevation_m'] + np.random.normal(0, 1.5, len(df)), 1)

# Slope: +/- ~0.3 degrees of "measurement drift"
df['Slope_deg'] = np.round(df['Slope_deg'] + np.random.normal(0, 0.3, len(df)), 1)

# Aspect: +/- ~1.2 degrees of "compass variance"
df['Aspect_deg'] = np.round(df['Aspect_deg'] + np.random.normal(0, 1.2, len(df)), 1)

# NDVI: +/- ~0.015 of "daily atmospheric/cloud noise"
# (We also use np.clip to ensure the NDVI never accidentally goes above 1.0 or below -1.0)
df['Baseline_NDVI'] = np.round(np.clip(df['Baseline_NDVI'] + np.random.normal(0, 0.015, len(df)), -1.0, 1.0), 4)


print("\n--- Applying Realistic Measurement Noise ---")
print("New 'Measured' Data (Notice the slight daily fluctuations):")
print(df[['Date', 'Elevation_m', 'Slope_deg', 'Baseline_NDVI']].head(10))

# 3. Save the new realistic dataset
output_file = 'ForestFire_8_Year_Realistic_Noise.csv'
df.to_csv(output_file, index=False)
print(f"\nSaved! Your realistic dataset is ready as '{output_file}'.")