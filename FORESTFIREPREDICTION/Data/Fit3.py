import pandas as pd
import numpy as np

# 1. Load your realistic 8-year dataset
filename = 'ForestFire_8_Year_Realistic_Noise.csv'
print(f"Loading raw data from {filename}...\n")
df = pd.read_csv(filename)

# Ensure Date is recognized as a time-series object and sort it properly
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Location_ID', 'Date']).reset_index(drop=True)

# ==========================================
# STEP 1: CREATE NEW PREDICTIVE FEATURES
# ==========================================
print("Generating new temporal and rolling features...")

# Extract the Month: This helps the algorithm naturally learn the 
# seasonal danger window in Uttarakhand (April - June).
df['Month'] = df['Date'].dt.month

# We must group by location so the rolling calculations don't bleed 
# weather data from one district into another.
grouped = df.groupby('Location_ID')

# The Build-Up Effect: Fires happen after prolonged dry heat.
# Calculate the total rain and average temp over the previous 7 days.
df['Rain_7d_Sum'] = grouped['Total_Rainfall_mm'].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
df['Temp_7d_Avg'] = grouped['Max_Temperature_C'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

# Custom Fire Danger Index: A single mathematical proxy for risk.
# High Temp * High Wind / Humidity (Added +1 to prevent division by zero)
df['Fire_Danger_Index'] = np.round((df['Max_Temperature_C'] * df['Max_Wind_Speed_kmh']) / (df['Max_Humidity_pct'] + 1), 2)


# ==========================================
# STEP 2: KEEP ONLY NECESSARY PARAMETERS
# ==========================================
print("Dropping identifiers and keeping only predictive features...")

# We drop human labels and coordinates to prevent the model from 
# simply memorizing locations instead of learning the environmental triggers.
columns_to_drop = ['Location_ID', 'District', 'Lat', 'Lon', 'Date']
ml_features_df = df.drop(columns=columns_to_drop)

# Reorder the final dataframe so it is clean and logical
final_columns = [
    # Engineered Features
    'Month', 'Rain_7d_Sum', 'Temp_7d_Avg', 'Fire_Danger_Index',
    # Daily Dynamic Weather
    'Max_Temperature_C', 'Max_Humidity_pct', 'Total_Rainfall_mm', 'Max_Wind_Speed_kmh',
    # Static Environment
    'Elevation_m', 'Slope_deg', 'Aspect_deg', 'Baseline_NDVI'
]
ml_features_df = ml_features_df[final_columns]

# ==========================================
# STEP 3: EXPORT
# ==========================================
print("\n--- Feature Engineering Complete ---")
print("Final Matrix Shape:", ml_features_df.shape)
print("\nColumns kept for the ML Model:")
for col in ml_features_df.columns:
    print(f" - {col}")

output_file = 'Engineered_Fire_Features.csv'
ml_features_df.to_csv(output_file, index=False)
print(f"\nSuccessfully saved clean dataset as '{output_file}'.")