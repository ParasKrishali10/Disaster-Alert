import pandas as pd
import numpy as np

# ==========================================
# 1. Load the Augmented Dataset
# ==========================================
input_file = 'Landslide_Augmented_Dataset.csv'
print(f"Loading {input_file}...")
df = pd.read_csv(input_file)

# ==========================================
# 2. Feature Engineering (Adding Physics & Math)
# ==========================================
print("Engineering new features...")

# A. Temporal Features (Seasonality / Monsoon tracking)
# Convert Date string to actual datetime object to extract the month
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
df['Month'] = df['Date'].dt.month

# Cyclical Encoding for Month (1 to 12)
df['Month_Sin'] = np.sin((df['Month'] - 1) * (2. * np.pi / 12))
df['Month_Cos'] = np.cos((df['Month'] - 1) * (2. * np.pi / 12))

# B. Cyclical Encoding for Aspect (0 to 360 degrees)
df['Aspect_Sin'] = np.sin(df['Aspect_deg'] * (np.pi / 180))
df['Aspect_Cos'] = np.cos(df['Aspect_deg'] * (np.pi / 180))

# C. Soil Moisture Gradient (Surface vs Deep)
# High positive number means the surface is heavily saturated compared to deep soil
df['Soil_Moisture_Gradient'] = df['Soil_Moisture_Surface'] - df['Soil_Moisture_Deep']

# D. Topo-Hydrology Interaction (Gravity + Water)
df['Rain_Slope_Interaction'] = df['Rainfall_Day_0_mm'] * df['Slope_deg']

# E. Total Hydrological Load
df['Total_15D_Water_Load'] = df['Rainfall_Antecedent_15D_mm'] + df['Rainfall_Day_0_mm']


# ==========================================
# 3. Drop Unnecessary & Redundant Columns
# ==========================================
print("Dropping identifiers and redundant columns...")

# We drop Lat/Lon to prevent spatial memorization.
# We drop Date because models can't read strings.
# We drop Month and Aspect_deg because we replaced them with Sine/Cosine.
columns_to_drop = ['Date', 'Lat', 'Lon', 'Month', 'Aspect_deg']

# Drop them from the dataframe
df = df.drop(columns=columns_to_drop)


# ==========================================
# 4. Save the Final ML-Ready Dataset
# ==========================================
output_file = 'Landslide_ML_Ready_Dataset.csv'
df.to_csv(output_file, index=False)

print("-" * 30)
print(f"SUCCESS! Dataset is ready for XGBoost.")
print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Saved as: {output_file}")
print("-" * 30)
print("Final Columns:")
print(list(df.columns))