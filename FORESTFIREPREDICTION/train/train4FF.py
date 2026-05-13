import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. LOAD DATA (Use the new 80%+ accuracy version)
# ==========================================
filename = 'Final_Forest_Fire_Data.csv'
print(f"Loading data from {filename}...")
df = pd.read_csv(filename)

# Define Features (X) and Target (y)
X = df.drop(columns=['Risk_Category', 'Risk_Level_Numeric'])
y = df['Risk_Level_Numeric']

# Split 80% Training / 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 2. INITIALIZE HYBRID COMPONENTS
# ==========================================
print("Initializing Hybrid components...")

# Model 1: Random Forest
rf_comp = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)

# Model 2: XGBoost
xgb_comp = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)

# Model 3: MLP (Needs a Pipeline for automated scaling)
mlp_pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
])

# ==========================================
# 3. CREATE THE TRIPLE HYBRID (VOTING)
# ==========================================
# 'soft' voting averages the probabilities of all three models
hybrid = VotingClassifier(
    estimators=[
        ('rf', rf_comp), 
        ('xgb', xgb_comp), 
        ('mlp', mlp_pipe)
    ], 
    voting='soft'
)

print("Training Triple Hybrid Ensemble (RF + XGB + MLP)... This might take 2-3 minutes.")
hybrid.fit(X_train, y_train)

# ==========================================
# 4. FINAL ANALYSIS & SAVING
# ==========================================
y_pred = hybrid.predict(X_test)
target_names = ['Low Risk (0)', 'Moderate Risk (1)', 'High Risk (2)', 'Extreme Risk (3)']

print("\n" + "="*30)
print("TRIPLE HYBRID FINAL REPORT")
print("="*30)
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Hybrid Model: Final Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual Risk Level', fontsize=12)
plt.xlabel('Predicted Risk Level', fontsize=12)
plt.tight_layout()
plt.savefig('hybrid_final_confusion_matrix.png', dpi=300)
plt.show()

# Save the master model
joblib.dump(hybrid, 'forest_fire_hybrid_model.pkl')
print("\nSuccess! Master Hybrid model saved as 'forest_fire_hybrid_model.pkl'")