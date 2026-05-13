import pandas as pd

def extract_unique_locations(filepath):
    print(f"Loading data from {filepath}...")
    
    try:
        # Read the text file, separating by the pipe character '|'
        # skiprows=[1,2,3,4] bypasses the initial headers and dash lines
        df = pd.read_csv(filepath, sep='|', skiprows=[1, 2, 3, 4], engine='python')
        
        # Clean up the column names by removing extra spaces
        df.columns = [col.strip() for col in df.columns]
        
        # Force the 'Lat' and 'Lon' columns to be numbers. 
        # errors='coerce' turns any remaining dashes or text into 'NaN' (Not a Number)
        df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce')
        df['Lon'] = pd.to_numeric(df['Lon'], errors='coerce')
        
        # Drop any rows that don't have valid coordinates
        df = df.dropna(subset=['Lat', 'Lon'])
        
        # Extract just the Latitude and Longitude columns and remove duplicates
        unique_locations = df[['Lat', 'Lon']].drop_duplicates().reset_index(drop=True)
        
        return unique_locations

    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'. Make sure it is in the same folder.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# --- Run the Extraction ---
if __name__ == "__main__":
    file_name = 'LandSlide Events.txt'
    
    # Extract the locations
    locations_df = extract_unique_locations(file_name)
    
    if locations_df is not None:
        print(f"\nSuccessfully extracted {len(locations_df)} unique locations.")
        print("-" * 30)
        print(locations_df.head(10)) # Print the first 10 to verify
        print("-" * 30)
        
        # Save this clean list to a new CSV file for easy access later
        output_name = 'Unique_Landslide_Locations.csv'
        locations_df.to_csv(output_name, index=False)
        print(f"Saved clean locations to: {output_name}")