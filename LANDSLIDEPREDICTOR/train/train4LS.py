import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import xgboost as xgb

# -----------------------------
# CREATE FOLDERS
# -----------------------------
os.makedirs("models", exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("Landslide_Modified_Dataset.csv")

X = df.drop("Landslide_Level", axis=1)
y = df["Landslide_Level"]

# -----------------------------
# SCALE FEATURES
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
# RANDOM FOREST TUNING
# -----------------------------
rf = RandomForestClassifier()

rf_params = {
    "n_estimators":[100,200],
    "max_depth":[6,10,None]
}

rf_grid = GridSearchCV(rf, rf_params, cv=3, n_jobs=-1)
rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_

# -----------------------------
# XGBOOST TUNING
# -----------------------------
xgb_model = xgb.XGBClassifier(objective="multi:softmax", num_class=3)

xgb_params = {
    "n_estimators":[100,200],
    "max_depth":[4,6],
    "learning_rate":[0.05,0.1]
}

xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, n_jobs=-1)
xgb_grid.fit(X_train, y_train)

best_xgb = xgb_grid.best_estimator_

# -----------------------------
# SVM TUNING
# -----------------------------
svm = SVC()

svm_params = {
    "C":[1,10],
    "kernel":["rbf"],
    "gamma":["scale","auto"]
}

svm_grid = GridSearchCV(svm, svm_params, cv=3, n_jobs=-1)
svm_grid.fit(X_train, y_train)

best_svm = svm_grid.best_estimator_

# -----------------------------
# STACKING HYBRID MODEL
# -----------------------------
estimators = [
    ("rf", best_rf),
    ("xgb", best_xgb),
    ("svm", best_svm)
]

hybrid_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

# -----------------------------
# TRAIN HYBRID MODEL
# -----------------------------
hybrid_model.fit(X_train, y_train)

# -----------------------------
# PREDICTION
# -----------------------------
y_pred = hybrid_model.predict(X_test)

# -----------------------------
# EVALUATION
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nHybrid Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(hybrid_model, "models/hybrid_landslide_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nHybrid model saved at models/hybrid_landslide_model.pkl")