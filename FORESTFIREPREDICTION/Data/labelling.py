import pandas as pd

# 1. Load your engineered features dataset
# Ensure 'Engineered_Fire_Features.csv' is uploaded to your Colab workspace
df = pd.read_csv('Engineered_Fire_Features.csv')

# 2. Define the exact classification rulebook
def classify_fire_risk(row):
    # Tier 4: Extreme Risk (The 30-30-30 Crossover Rule)
    # Temp >= 30C, Humidity <= 30%, and Zero Rain in the last 7 days
    if (row['Max_Temperature_C'] >= 30) and (row['Max_Humidity_pct'] <= 30) and (row['Rain_7d_Sum'] == 0):
        return 'Extreme Risk'
    
    # Tier 3: High Risk
    # High Danger Index OR essentially no rain in the last week
    elif (row['Fire_Danger_Index'] > 12) or (row['Rain_7d_Sum'] < 2.0):
        return 'High Risk'
        
    # Tier 2: Moderate Risk
    # Warming up, but winds/temps are not in the danger zone yet
    elif (row['Fire_Danger_Index'] > 5) and (row['Fire_Danger_Index'] <= 12):
        return 'Moderate Risk'
        
    # Tier 1: Low Risk
    # Damp, high humidity, or recent rainfall
    else:
        return 'Low Risk'

print("Scanning daily weather and applying Fire Risk Classification...")

# 3. Apply the function to create your new Label column
df['Risk_Category'] = df.apply(classify_fire_risk, axis=1)

# Optional but recommended for Machine Learning: 
# Algorithms require numbers, not words. This creates a numeric version of your labels.
risk_mapping = {
    'Low Risk': 0,
    'Moderate Risk': 1,
    'High Risk': 2,
    'Extreme Risk': 3
}
df['Risk_Level_Numeric'] = df['Risk_Category'].map(risk_mapping)

# 4. Show the distribution of the newly labeled data
print("\n--- Final Dataset Class Distribution ---")
print(df['Risk_Category'].value_counts())

print("\nPreview of Labeled Data:")
print(df[['Temp_7d_Avg', 'Rain_7d_Sum', 'Fire_Danger_Index', 'Risk_Category', 'Risk_Level_Numeric']].head(10))

# 5. Save the final labeled dataset
output_filename = 'Labeled_Fire_Risk_Data.csv'
df.to_csv(output_filename, index=False)
print(f"\nSuccess! Your perfectly labeled dataset is saved as '{output_filename}' and ready for ML.")