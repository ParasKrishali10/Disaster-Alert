import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from imblearn.over_sampling import SMOTE

# -----------------------------
# 1. CREATE PLOTS FOLDER
# -----------------------------
os.makedirs("plots", exist_ok=True)

# -----------------------------
# 2. LOAD DATASET
# -----------------------------
df = pd.read_csv("Landslide_Labeled_Dataset.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# -----------------------------
# 3. BASIC EDA
# -----------------------------
print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# -----------------------------
# 4. CLASS DISTRIBUTION BEFORE
# -----------------------------
print("\nClass Distribution BEFORE:")
print(df["Landslide_Level"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="Landslide_Level", data=df)
plt.title("Class Distribution Before Processing")

plt.savefig("plots/class_distribution_before.png", dpi=300)
plt.close()

# -----------------------------
# 5. CORRELATION HEATMAP
# -----------------------------
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.savefig("plots/correlation_heatmap.png", dpi=300)
plt.close()

# -----------------------------
# 6. FEATURE DISTRIBUTION
# -----------------------------
plt.figure(figsize=(6,4))
sns.histplot(df["Rainfall_Antecedent_15D_mm"], bins=30, kde=True)

plt.title("Rainfall Distribution")

plt.savefig("plots/rainfall_distribution.png", dpi=300)
plt.close()

# -----------------------------
# 7. CREATE LEVEL 3 CASES
# -----------------------------
level3_condition = (
    (df["Rainfall_Antecedent_15D_mm"] > 180) &
    (df["Soil_Moisture_Surface"] > 0.75) &
    (df["Slope_deg"] > 40)
)

df.loc[level3_condition, "Landslide_Level"] = 3

print("\nClass Distribution AFTER Level 3 Creation:")
print(df["Landslide_Level"].value_counts())

# -----------------------------
# 8. SPLIT FEATURES / TARGET
# -----------------------------
X = df.drop("Landslide_Level", axis=1)
y = df["Landslide_Level"]

# -----------------------------
# 9. APPLY SMOTE
# -----------------------------
smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)

df_balanced = pd.concat(
    [pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)],
    axis=1
)

df_balanced.columns = df.columns

print("\nClass Distribution AFTER SMOTE:")
print(df_balanced["Landslide_Level"].value_counts())

# -----------------------------
# 10. PLOT AFTER SMOTE
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Landslide_Level", data=df_balanced)

plt.title("Class Distribution After SMOTE")

plt.savefig("plots/class_distribution_after_smote.png", dpi=300)
plt.close()

# -----------------------------
# 11. SAVE BALANCED DATASET
# -----------------------------
df_balanced.to_csv("Landslide_Balanced_Dataset.csv", index=False)

print("\nBalanced dataset saved as Landslide_Balanced_Dataset.csv")
print("All plots saved inside the 'plots' folder.")