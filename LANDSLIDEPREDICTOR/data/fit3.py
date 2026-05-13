import pandas as pd

# load dataset
df = pd.read_csv("Landslide_Unlabeled_Dataset.csv")

# normalize important variables
df["Rain_norm"] = df["Rainfall_Antecedent_15D_mm"] / df["Rainfall_Antecedent_15D_mm"].max()
df["Slope_norm"] = df["Slope_deg"] / df["Slope_deg"].max()
df["Moisture_norm"] = df["Soil_Moisture_Surface"] / df["Soil_Moisture_Surface"].max()

# compute risk score
df["Risk_Score"] = (
    0.4 * df["Rain_norm"] +
    0.3 * df["Slope_norm"] +
    0.3 * df["Moisture_norm"]
) * 100


# classification function
def classify_risk(score):
    if score < 30:
        return 0   # Green
    elif score < 60:
        return 1   # Yellow
    else :
        return 2   # Red

# apply classification
df["Landslide_Level"] = df["Risk_Score"].apply(classify_risk)

# save labeled dataset
df.to_csv("Landslide_Labeled_Dataset.csv", index=False)

print("Dataset labeled successfully.")
print(df[["Risk_Score","Landslide_Level"]].head())