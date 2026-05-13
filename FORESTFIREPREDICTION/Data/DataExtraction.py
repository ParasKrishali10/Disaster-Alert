import ee
import pandas as pd
import numpy as np
import io
import requests
import time
from datetime import datetime, timedelta

# ==========================================
# PHASE 1: GOOGLE EARTH ENGINE EXTRACTION
# ==========================================
print("--- PHASE 1: Initializing Earth Engine & Extracting Spatial Data ---")
# Make sure you have run ee.Authenticate() once in your Colab session before this!
ee.Initialize(project='lofty-inn-490212-m9')

# Clean, embedded CSV data for your 10 locations
csv_data = """ID,Date,Time,Location,District,Lat,Lon
1,12-05-2009,14:30,Lansdowne forest,Pauri Garhwal,29.840,78.683
2,18-05-2009,13:20,Almora ridge forest,Almora,29.597,79.659
3,03-06-2010,15:10,Ramnagar forest division,Nainital,29.392,79.128
4,14-05-2011,12:40,Champawat hills,Champawat,29.336,80.091
7,19-05-2013,14:50,Pithoragarh ridge forest,Pithoragarh,29.583,80.218
9,02-05-2014,14:10,Tehri forest range,Tehri,30.375,78.480
14,19-04-2016,16:00,Bageshwar pine forests,Bageshwar,29.838,79.771
15,29-04-2016,12:50,Chamoli forest slopes,Chamoli,30.404,79.320
16,04-05-2016,14:40,Rudraprayag forests,Rudraprayag,30.284,78.980
19,15-05-2016,14:15,Uttarkashi forest slopes,Uttarkashi,30.729,78.443
"""
base_df = pd.read_csv(io.StringIO(csv_data))

srtm = ee.Image('USGS/SRTMGL1_003')
terrain = ee.Terrain.products(srtm)

def get_gee_features(lat, lon, date_str):
    try:
        point = ee.Geometry.Point([lon, lat])
        fire_date = datetime.strptime(date_str, "%d-%m-%Y")
        start_date = fire_date - timedelta(days=30)
        
        ee_start = start_date.strftime("%Y-%m-%d")
        ee_end = fire_date.strftime("%Y-%m-%d")

        terrain_data = terrain.reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=30).getInfo()
        
        ndvi_collection = (ee.ImageCollection('MODIS/061/MOD13Q1')
                           .filterBounds(point).filterDate(ee_start, ee_end).select('NDVI'))
        ndvi_data = ndvi_collection.median().reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=250).getInfo()
        
        ndvi_val = (ndvi_data.get('NDVI') * 0.0001) if ndvi_data and ndvi_data.get('NDVI') else None
        
        return {
            'Elevation_m': terrain_data.get('elevation'),
            'Slope_deg': terrain_data.get('slope'),
            'Aspect_deg': terrain_data.get('aspect'),
            'Pre_Fire_NDVI': ndvi_val
        }
    except Exception as e:
        print(f"  -> GEE Error for {lat}, {lon}: {e}")
        return {'Elevation_m': None, 'Slope_deg': None, 'Aspect_deg': None, 'Pre_Fire_NDVI': None}

gee_results = []
for index, row in base_df.iterrows():
    print(f"  Fetching spatial data for {row['District']}...")
    features = get_gee_features(row['Lat'], row['Lon'], row['Date'])
    features.update(row.to_dict()) # Combine original row data
    gee_results.append(features)
    time.sleep(0.5)

spatial_df = pd.DataFrame(gee_results)


# ==========================================
# PHASE 2: OPEN-METEO 8-YEAR EXTRACTION
# ==========================================
print("\n--- PHASE 2: Fetching 8-Year Historical Weather Data ---")

START_DATE = "2018-01-01"
END_DATE = "2025-12-31"

def get_8_years_daily_weather(lat, lon):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": "temperature_2m_max,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_max",
        "timezone": "Asia/Kolkata"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame({
            'Date': data['daily']['time'],
            'Max_Temperature_C': data['daily']['temperature_2m_max'],
            'Max_Humidity_pct': data['daily']['relative_humidity_2m_max'],
            'Total_Rainfall_mm': data['daily']['precipitation_sum'],
            'Max_Wind_Speed_kmh': data['daily']['wind_speed_10m_max']
        })
    except Exception as e:
        print(f"  -> Weather Error for {lat}, {lon}: {e}")
        return pd.DataFrame()

all_locations_data = []
for index, row in spatial_df.iterrows():
    print(f"  Pulling 2,920 days of weather for {row['District']}...")
    weather_timeline = get_8_years_daily_weather(row['Lat'], row['Lon'])
    
    if not weather_timeline.empty:
        # Broadcast the spatial data across the 8-year timeline
        weather_timeline['Location_ID'] = row['ID']
        weather_timeline['District'] = row['District']
        weather_timeline['Lat'] = row['Lat']
        weather_timeline['Lon'] = row['Lon']
        weather_timeline['Elevation_m'] = row['Elevation_m']
        weather_timeline['Slope_deg'] = row['Slope_deg']
        weather_timeline['Aspect_deg'] = row['Aspect_deg']
        weather_timeline['Baseline_NDVI'] = row['Pre_Fire_NDVI'] 
        all_locations_data.append(weather_timeline)
        
    time.sleep(1)

master_df = pd.concat(all_locations_data, ignore_index=True)


# ==========================================
# PHASE 3: REALISTIC NOISE INJECTION
# ==========================================
print("\n--- PHASE 3: Applying Realistic Sensor Noise ---")
np.random.seed(42)

master_df['Elevation_m'] = np.round(master_df['Elevation_m'] + np.random.normal(0, 1.5, len(master_df)), 1)
master_df['Slope_deg'] = np.round(master_df['Slope_deg'] + np.random.normal(0, 0.3, len(master_df)), 1)
master_df['Aspect_deg'] = np.round(master_df['Aspect_deg'] + np.random.normal(0, 1.2, len(master_df)), 1)
master_df['Baseline_NDVI'] = np.round(np.clip(master_df['Baseline_NDVI'] + np.random.normal(0, 0.015, len(master_df)), -1.0, 1.0), 4)

# Reorder columns for clean ML processing
columns_order = [
    'Date', 'Location_ID', 'District', 'Lat', 'Lon', 
    'Max_Temperature_C', 'Max_Humidity_pct', 'Total_Rainfall_mm', 'Max_Wind_Speed_kmh', 
    'Elevation_m', 'Slope_deg', 'Aspect_deg', 'Baseline_NDVI'
]
final_master_df = master_df[columns_order]

print("\n--- Pipeline Complete! ---")
print(f"Total Rows Generated: {len(final_master_df)}")

# Save the final masterpiece
output_filename = 'ForestFire_Final_Pipeline_Data.csv'
final_master_df.to_csv(output_filename, index=False)
print(f"\nSuccess! The ultimate dataset is saved as '{output_filename}'.")