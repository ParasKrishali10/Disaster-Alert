import pandas as pd
import numpy as np

# ==========================================
# 1. Load the Datasets
# ==========================================
massive_df = pd.read_csv('Landslide_Massive_Dataset.csv')
gee_df = pd.read_csv('GEE_Extracted_Features.csv')

# Round coordinates to 3 decimal places to ensure a perfect merge
massive_df['Lat_round'] = massive_df['Lat'].round(3)
massive_df['Lon_round'] = massive_df['Lon'].round(3)
gee_df['Lat_round'] = gee_df['Lat'].round(3)
gee_df['Lon_round'] = gee_df['Lon'].round(3)

# ==========================================
# 2. Merge Weather and Topography
# ==========================================
merged = pd.merge(massive_df, gee_df, on=['Lat_round', 'Lon_round'], how='left', suffixes=('', '_gee'))

# Prefer the GEE elevation over the Topo API one
if 'Elevation_m_gee' in merged.columns:
    merged['Elevation_m'] = merged['Elevation_m_gee']
    
# Drop the redundant merging columns
cols_to_drop = ['Lat_gee', 'Lon_gee', 'Lat_round', 'Lon_round', 'Elevation_m_gee']
merged = merged.drop(columns=[c for c in cols_to_drop if c in merged.columns])

# ==========================================
# 3. Apply Data Augmentation (Spatial Jitter)
# ==========================================
print("Applying spatial jitter to create unique location profiles...")
np.random.seed(42) # Keeps the randomness consistent if you re-run it
n_rows = len(merged)

# 1. Jitter Coordinates (adds approx +/- 15 meters of variation)
merged['Lat'] = merged['Lat'] + np.random.uniform(-0.00015, 0.00015, n_rows)
merged['Lon'] = merged['Lon'] + np.random.uniform(-0.00015, 0.00015, n_rows)

# 2. Jitter Elevation (+/- 5 meters)
if 'Elevation_m' in merged.columns:
    merged['Elevation_m'] = merged['Elevation_m'] + np.random.uniform(-5.0, 5.0, n_rows)

# 3. Jitter Slope (+/- 2.5 degrees, clipped so it doesn't go below 0)
if 'Slope_deg' in merged.columns:
    merged['Slope_deg'] = merged['Slope_deg'] + np.random.uniform(-2.5, 2.5, n_rows)
    merged['Slope_deg'] = np.clip(merged['Slope_deg'], 0, 90)

# 4. Jitter Aspect (+/- 10 degrees, wrapped around a 360-degree compass)
if 'Aspect_deg' in merged.columns:
    merged['Aspect_deg'] = (merged['Aspect_deg'] + np.random.uniform(-10, 10, n_rows)) % 360

# 5. Jitter Vegetation/NDVI (+/- 0.03, ensuring it stays between -1 and 1)
if 'Baseline_NDVI' in merged.columns:
    # Fill any missing NDVI values with the median before augmenting
    merged['Baseline_NDVI'] = merged['Baseline_NDVI'].fillna(merged['Baseline_NDVI'].median())
    merged['Baseline_NDVI'] = merged['Baseline_NDVI'] + np.random.uniform(-0.03, 0.03, n_rows)
    merged['Baseline_NDVI'] = np.clip(merged['Baseline_NDVI'], -1.0, 1.0)

# ==========================================
# 4. Final Cleanup and Save
# ==========================================
# Reorder the columns so they are clean and ready for XGBoost
final_cols = ['Target', 'Date', 'Lat', 'Lon', 'Elevation_m', 'Slope_deg', 'Aspect_deg', 'Baseline_NDVI',
              'Rainfall_Day_0_mm', 'Rainfall_Antecedent_3D_mm', 'Rainfall_Antecedent_7D_mm', 
              'Rainfall_Antecedent_15D_mm', 'Soil_Moisture_Surface', 'Soil_Moisture_Deep']

# Keep only the columns that actually exist to prevent errors
final_cols = [c for c in final_cols if c in merged.columns]
final_df = merged[final_cols]

output_filename = "Landslide_Augmented_Dataset.csv"
final_df.to_csv(output_filename, index=False)

print("-" * 30)
print("AUGMENTATION COMPLETE!")
print(f"Final dataset shape: {final_df.shape}")
print(f"Saved to: {output_filename}")