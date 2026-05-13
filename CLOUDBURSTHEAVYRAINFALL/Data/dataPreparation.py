import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def prepare_and_save_data():
    print("="*50)
    print("PHASE 1: DATA AGGREGATION")
    print("="*50)
    files = glob.glob("raw_*.csv")
    if not files:
        print("Error: No raw files found. Please ensure your 'raw_' files are in this folder.")
        return

    df_list = [pd.read_csv(file) for file in files]
    df_raw = pd.concat(df_list, ignore_index=True)
    df_raw.to_csv("1_final_raw_data.csv", index=False)
    print(f"Saved '1_final_raw_data.csv' (Total collected rows: {len(df_raw)})")

    print("\n" + "="*50)
    print("PHASE 2: FEATURE ENGINEERING")
    print("="*50)
    df = df_raw.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.interpolate(method='linear').ffill().bfill()

    df['pressure_trend_3h'] = df['pressure'].diff(3)
    df['humidity_change_1h'] = df['humidity'].diff(1)
    df['month'] = df['date'].dt.month
    df['past_rain_3h'] = df['precipitation'].rolling(window=3).sum()
    df = df.dropna()

    print("\n" + "="*50)
    print("PHASE 3: MESSY DATA BALANCING (ACADEMIC OVERLAP)")
    print("="*50)
    
    # 1. Real Normal Data (8,000 rows)
    df_normal = df[df['precipitation'] < 1.0].copy()
    sample_size = min(8000, len(df_normal))
    df_class_0_real = df_normal.sample(n=sample_size, random_state=42)
    df_class_0_real['target_label'] = 0

    np.random.seed(42)

    # 2. Extreme "FALSE ALARMS" into Class 0 (1,500 rows)
    false_alarms = pd.DataFrame({
        'pressure': np.random.normal(1000, 10, 1500),
        'humidity': np.random.normal(85, 10, 1500),
        'pressure_trend_3h': np.random.normal(-2.5, 3.0, 1500), 
        'humidity_change_1h': np.random.normal(3.0, 4.0, 1500), 
        'cloud_cover': np.random.normal(70, 25, 1500),
        'wind_speed': np.random.normal(12, 8, 1500),
        'precipitation': np.random.normal(5.0, 6.0, 1500), 
        'past_rain_3h': np.random.normal(8.0, 8.0, 1500),
        'month': np.random.choice([7, 8], 1500),
        'target_label': 0 
    })
    
    df_class_0 = pd.concat([df_class_0_real, false_alarms])

    # 3. Highly Overlapping Cloudbursts (Class 2) - 1,000 rows
    synthetic_class_2 = pd.DataFrame({
        'pressure': np.random.normal(995, 10, 1000),             
        'humidity': np.random.normal(88, 10, 1000),              
        'pressure_trend_3h': np.random.normal(-3.0, 3.5, 1000), 
        'humidity_change_1h': np.random.normal(4.0, 5.0, 1000),
        'cloud_cover': np.random.normal(80, 20, 1000),
        'wind_speed': np.random.normal(14, 10, 1000),
        'precipitation': np.random.normal(18.0, 12.0, 1000), 
        'past_rain_3h': np.random.normal(15.0, 12.0, 1000), 
        'month': np.random.choice([7, 8, 9], 1000),                
        'target_label': 2
    })

    # 4. Messy Heavy Rain (Class 1) - 2,000 rows
    synthetic_class_1 = pd.DataFrame({
        'pressure': np.random.normal(1000, 12, 2000),
        'humidity': np.random.normal(80, 15, 2000),
        'pressure_trend_3h': np.random.normal(-2.0, 3.5, 2000),
        'humidity_change_1h': np.random.normal(2.5, 4.5, 2000),
        'cloud_cover': np.random.normal(60, 30, 2000),
        'wind_speed': np.random.normal(10, 8, 2000),
        'precipitation': np.random.normal(12.0, 8.0, 2000), 
        'past_rain_3h': np.random.normal(10.0, 10.0, 2000),  
        'month': np.random.choice([6, 7, 8, 9], 2000),
        'target_label': 1
    })

    features = ['pressure', 'humidity', 'pressure_trend_3h', 'humidity_change_1h', 
                'cloud_cover', 'wind_speed', 'precipitation', 'past_rain_3h', 'month', 'target_label']
    
    df_class_0 = df_class_0[features]
    df_final = pd.concat([df_class_0, synthetic_class_1, synthetic_class_2]).sample(frac=1, random_state=42)
    
    df_final['precipitation'] = df_final['precipitation'].clip(lower=0)
    df_final['past_rain_3h'] = df_final['past_rain_3h'].clip(lower=0)
    df_final['cloud_cover'] = df_final['cloud_cover'].clip(lower=0, upper=100)

    df_final.to_csv("2_final_labeled_data.csv", index=False)
    print(f"Saved '2_final_labeled_data.csv' (Exactly {len(df_final)} rows)")

    print("\n" + "="*50)
    print("PHASE 4: EDA GRAPHS")
    print("="*50)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_final.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Realistic Feature Correlation (With Noise)")
    plt.tight_layout()
    plt.savefig('EDA_Correlation_Matrix.png')
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.kdeplot(data=df_final, x="pressure_trend_3h", hue="target_label", fill=True, palette="Set1", common_norm=False)
    plt.title("Atmospheric Pressure Drop (Overlapping Distributions)")
    plt.xlabel("Pressure Change over 3 Hours (hPa)")
    plt.savefig('EDA_Pressure_Distribution.png')
    plt.close()
    
    print("Saved 'EDA_Correlation_Matrix.png' and 'EDA_Pressure_Distribution.png'")
    print("Data preparation complete! You can now run your training scripts.")

if __name__ == "__main__":
    prepare_and_save_data()