import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load and Split
df = pd.read_csv('Final_Forest_Fire_Data.csv')
X = df.drop(columns=['Risk_Category', 'Risk_Level_Numeric'])
y = df['Risk_Level_Numeric']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Analysis
y_pred = rf.predict(X_test)
print("--- RANDOM FOREST ANALYSIS ---")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.savefig('rf_confusion_matrix.png')

# Feature Importance
fi = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=fi)
plt.title('Random Forest Feature Importance')
plt.savefig('rf_feature_importance.png')

joblib.dump(rf, 'forest_fire_rf_model.pkl')