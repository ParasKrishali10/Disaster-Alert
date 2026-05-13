import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load, Split, and Scale
df = pd.read_csv('Final_Forest_Fire_Data.csv')
X = df.drop(columns=['Risk_Category', 'Risk_Level_Numeric'])
y = df['Risk_Level_Numeric']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Analysis
y_pred = mlp.predict(X_test_scaled)
print("--- MLP NEURAL NETWORK ANALYSIS ---")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Purples')
plt.title('MLP Confusion Matrix')
plt.savefig('mlp_confusion_matrix.png')

joblib.dump(mlp, 'forest_fire_mlp_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')