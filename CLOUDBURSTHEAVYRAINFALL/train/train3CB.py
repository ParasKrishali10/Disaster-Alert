import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

def train_svm():
    print("Loading preprocessed dataset...")
    df = pd.read_csv("2_final_labeled_data.csv")
    X = df.drop(columns=['target_label'])
    y = df['target_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training SVM Pipeline...")
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))
    ])
    svm_pipeline.fit(X_train, y_train)
    y_pred = svm_pipeline.predict(X_test)
    
    target_names = ['Normal', 'Heavy Rain', 'Cloudburst']
    print("\n--- SVM REPORT ---")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Support Vector Machine Confusion Matrix')
    plt.ylabel('Actual Event')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig('SVM_Confusion_Matrix.png')
    plt.close()

    # 2. Feature Importance (Permutation)
    print("Calculating SVM Permutation Feature Importance (This takes a few seconds)...")
    result = permutation_importance(svm_pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    importances = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar', color='royalblue')
    plt.title('SVM Feature Importance (Permutation Method)')
    plt.tight_layout()
    plt.savefig('SVM_Feature_Importance.png')
    plt.close()

    with open('model_svm.pkl', 'wb') as f:
        pickle.dump(svm_pipeline, f)
    print("Saved 'SVM_Confusion_Matrix.png', 'SVM_Feature_Importance.png', and 'model_svm.pkl'")

if __name__ == "__main__":
    train_svm()