import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load and Split
df = pd.read_csv('Final_Forest_Fire_Data.csv')
X = df.drop(columns=['Risk_Category', 'Risk_Level_Numeric'])
y = df['Risk_Level_Numeric']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
xgb.fit(X_train, y_train)

# Analysis
y_pred = xgb.predict(X_test)
print("--- XGBOOST ANALYSIS ---")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.title('XGBoost Confusion Matrix')
plt.savefig('xgb_confusion_matrix.png')

# Feature Importance
fi_xgb = pd.DataFrame({'Feature': X.columns, 'Importance': xgb.feature_importances_}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=fi_xgb)
plt.title('XGBoost Feature Importance')
plt.savefig('xgb_feature_importance.png')

joblib.dump(xgb, 'forest_fire_xgb_model.pkl')