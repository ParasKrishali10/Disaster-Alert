import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# CREATE OUTPUT FOLDERS
# -----------------------------
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("Landslide_Modified_Dataset.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

# -----------------------------
# BASIC ANALYSIS
# -----------------------------
print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nClass Distribution:")
print(df["Landslide_Level"].value_counts())

# -----------------------------
# FEATURE / TARGET SPLIT
# -----------------------------
X = df.drop("Landslide_Level", axis=1)
y = df["Landslide_Level"]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# -----------------------------
# RANDOM FOREST MODEL
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

# -----------------------------
# PREDICTION
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# MODEL EVALUATION
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("plots/confusion_matrix_rf.png", dpi=300)
plt.close()

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=importance_df.head(10))

plt.title("Top Feature Importance (Random Forest)")

plt.savefig("plots/feature_importance_rf.png", dpi=300)
plt.close()

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "models/random_forest_landslide_model.pkl")

print("\nModel saved at: models/random_forest_landslide_model.pkl")
print("\nPlots saved in 'plots/' folder")