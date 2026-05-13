import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Landslide_Balanced_Dataset.csv")

print("Original class distribution:")
print(df["Landslide_Level"].value_counts())

# -----------------------------
# REMOVE CLASS 3
# -----------------------------
df = df[df["Landslide_Level"] != 3]

print("\nAfter removing class 3:")
print(df["Landslide_Level"].value_counts())


# -----------------------------
# ADD FEATURE NOISE
# -----------------------------
noise_factor = 0.05

feature_cols = df.columns.drop("Landslide_Level")

for col in feature_cols:
    df[col] = df[col] + np.random.normal(0, noise_factor, size=len(df))


# -----------------------------
# ADD RANDOM FEATURE
# -----------------------------
df["random_noise_feature"] = np.random.rand(len(df))


# -----------------------------
# RANDOMLY FLIP LABELS
# -----------------------------
flip_ratio = 0.12

num_flip = int(len(df) * flip_ratio)

flip_indices = np.random.choice(df.index, num_flip, replace=False)

possible_classes = [0,1,2]  # NO CLASS 3

for idx in flip_indices:
    current_label = df.loc[idx, "Landslide_Level"]
    new_label = np.random.choice([c for c in possible_classes if c != current_label])
    df.loc[idx, "Landslide_Level"] = new_label


print("\nModified class distribution:")
print(df["Landslide_Level"].value_counts())


# -----------------------------
# SAVE DATASET
# -----------------------------
df.to_csv("Landslide_Modified_Dataset.csv", index=False)

print("\nModified dataset saved as Landslide_Modified_Dataset.csv")