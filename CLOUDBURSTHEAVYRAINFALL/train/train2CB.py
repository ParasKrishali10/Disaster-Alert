import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_rf():
    print("Loading preprocessed dataset...")
    df = pd.read_csv("2_final_labeled_data.csv")
    X = df.drop(columns=['target_label'])
    y = df['target_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    target_names = ['Normal', 'Heavy Rain', 'Cloudburst']
    print("\n--- RANDOM FOREST REPORT ---")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
    plt.title('Random Forest Confusion Matrix')
    plt.ylabel('Actual Event')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig('RF_Confusion_Matrix.png')
    plt.close()

    # 2. Feature Importance (Native)
    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar', color='forestgreen')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig('RF_Feature_Importance.png')
    plt.close()

    with open('model_rf.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("Saved 'RF_Confusion_Matrix.png', 'RF_Feature_Importance.png', and 'model_rf.pkl'")

if __name__ == "__main__":
    train_rf()