import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
# XGBOOST MODEL
# -----------------------------
model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=4,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
print("\nTraining XGBoost model...")
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

plt.savefig("plots/confusion_matrix.png", dpi=300)
plt.close()

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
plt.figure(figsize=(10,6))

xgb.plot_importance(model, max_num_features=10)

plt.title("Feature Importance")

plt.savefig("plots/feature_importance.png", dpi=300)
plt.close()

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "models/xgboost_landslide_model.pkl")

print("\nModel saved at: models/xgboost_landslide_model.pkl")

print("\nPlots saved in 'plots/' folder")