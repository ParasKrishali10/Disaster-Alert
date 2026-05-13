import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

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
# FEATURE SCALING (important for SVM)
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# SVM MODEL
# -----------------------------
model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
print("\nTraining SVM model...")
model.fit(X_train, y_train)

# -----------------------------
# PREDICTIONS
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
plt.title("Confusion Matrix (SVM)")

plt.savefig("plots/confusion_matrix_svm.png", dpi=300)
plt.close()

# -----------------------------
# SAVE MODEL + SCALER
# -----------------------------
joblib.dump(model, "models/svm_landslide_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel saved at: models/svm_landslide_model.pkl")
print("Scaler saved at: models/scaler.pkl")
print("\nPlots saved in 'plots/' folder")