import pandas as pd
import requests
import time
from datetime import datetime, timedelta

# ==========================================
# 1. Load the Text File
# ==========================================
print("Loading data from LandSlide Events.txt...")
try:
    df = pd.read_csv('LandSlide Events.txt', sep='|', skiprows=[1, 2, 3, 4])
    df.columns = [col.strip() for col in df.columns]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce')
    df['Lon'] = pd.to_numeric(df['Lon'], errors='coerce')
    df = df.dropna(subset=['Lat', 'Lon']).reset_index(drop=True)
    print(f"Loaded {len(df)} locations.")
except Exception as e:
    print(f"File error: {e}")
    exit()

# ==========================================
# 2. Bulk API Fetching Functions
# ==========================================
def get_elevation(lat, lon):
    url = f"https://api.opentopodata.org/v1/srtm30m?locations={lat},{lon}"
    try:
        res = requests.get(url, timeout=10).json()
        return res['results'][0]['elevation']
    except:
        return None

def get_bulk_weather(lat, lon, landslide_date_str):
    """Pulls a continuous 2-year block of daily weather in one API call."""
    event_date = datetime.strptime(landslide_date_str.strip(), '%d-%m-%Y')
    
    # Define a 730-day window (1 year before, 1 year after)
    start_date = event_date - timedelta(days=365)
    end_date = event_date + timedelta(days=365)
    
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date.strftime('%Y-%m-%d')}&end_date={end_date.strftime('%Y-%m-%d')}&daily=precipitation_sum,soil_moisture_0_to_7cm_mean,soil_moisture_7_to_28cm_mean&timezone=auto"
    
    try:
        res = requests.get(url, timeout=15).json()
        
        # Load the massive daily array directly into Pandas
        df_mini = pd.DataFrame({
            'Date': pd.to_datetime(res['daily']['time']),
            'Rainfall_Day_0_mm': res['daily']['precipitation_sum'],
            'Soil_Moisture_Surface': res['daily']['soil_moisture_0_to_7cm_mean'],
            'Soil_Moisture_Deep': res['daily']['soil_moisture_7_to_28cm_mean']
        })
        
        # Calculate antecedent rainfall instantly using rolling windows
        # .shift(1) ensures we look at the days *before* the current row
        df_mini['Rainfall_Antecedent_3D_mm'] = df_mini['Rainfall_Day_0_mm'].shift(1).rolling(window=3).sum()
        df_mini['Rainfall_Antecedent_7D_mm'] = df_mini['Rainfall_Day_0_mm'].shift(1).rolling(window=7).sum()
        df_mini['Rainfall_Antecedent_15D_mm'] = df_mini['Rainfall_Day_0_mm'].shift(1).rolling(window=15).sum()
        
        # Set Target = 1 ONLY on the exact day of the landslide, otherwise 0
        df_mini['Target'] = (df_mini['Date'] == event_date).astype(int)
        
        # Drop the first 15 rows because they don't have enough history for the 15-day sum
        df_mini = df_mini.dropna()
        
        return df_mini
    except Exception as e:
        print(f"API Error: {e}")
        return None

# ==========================================
# 3. Main Processing Loop
# ==========================================
all_dataframes = []

print("\nExecuting bulk 2-year extraction per location...")
for index, row in df.iterrows():
    lat, lon, date_str = row['Lat'], row['Lon'], row['Date']
    print(f"[{index+1}/{len(df)}] Pulling 700+ days of data for {lat}, {lon}...")
    
    elevation = get_elevation(lat, lon)
    bulk_df = get_bulk_weather(lat, lon, date_str)
    
    if bulk_df is not None:
        # Add static data to the block
        bulk_df['Lat'] = lat
        bulk_df['Lon'] = lon
        bulk_df['Elevation_m'] = elevation
        
        # Format date back to string to match previous formats
        bulk_df['Date'] = bulk_df['Date'].dt.strftime('%d-%m-%Y')
        
        all_dataframes.append(bulk_df)
    
    time.sleep(1.5) # Avoid API rate limits

# ==========================================
# 4. Combine and Save
# ==========================================
final_dataset = pd.concat(all_dataframes, ignore_index=True)

# Reorder columns so Target is first
cols = ['Target', 'Date', 'Lat', 'Lon', 'Elevation_m', 'Rainfall_Day_0_mm', 
        'Rainfall_Antecedent_3D_mm', 'Rainfall_Antecedent_7D_mm', 
        'Rainfall_Antecedent_15D_mm', 'Soil_Moisture_Surface', 'Soil_Moisture_Deep']
final_dataset = final_dataset[cols]

final_dataset.to_csv("Landslide_Massive_Dataset.csv", index=False)

print("-" * 30)
print("EXTRACTION COMPLETE!")
print(f"Total rows generated: {len(final_dataset)}")
print("File saved as: Landslide_Massive_Dataset.csv")