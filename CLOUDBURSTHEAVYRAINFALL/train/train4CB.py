import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

def train_tuned_hybrid():
    print("Loading preprocessed dataset...")
    df = pd.read_csv("2_final_labeled_data.csv")
    X = df.drop(columns=['target_label'])
    y = df['target_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Initializing base models...")
    xgb_base = xgb.XGBClassifier(objective='multi:softprob', num_class=3, random_state=42, n_jobs=-1)
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    svm_base = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))])

    hybrid_model = VotingClassifier(estimators=[('xgb', xgb_base), ('rf', rf_base), ('svm', svm_base)], voting='soft')

    print("Running Hyperparameter Tuning (GridSearchCV)...")
    param_grid = {
        'xgb__max_depth': [3, 4],
        'xgb__learning_rate': [0.05, 0.1],
        'rf__max_depth': [3, 5],
        'weights': [[1, 1, 1], [2, 1, 1]] 
    }

    grid_search = GridSearchCV(estimator=hybrid_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_hybrid = grid_search.best_estimator_

    print("\nBest Parameters Found:", grid_search.best_params_)
    y_pred = best_hybrid.predict(X_test)
    
    target_names = ['Normal', 'Heavy Rain', 'Cloudburst']
    print("\n--- TUNED HYBRID REPORT ---")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=target_names, yticklabels=target_names)
    plt.title('Tuned Hybrid Model (XGB + RF + SVM)')
    plt.ylabel('Actual Event')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig('Hybrid_Confusion_Matrix.png')
    plt.close()

    # 2. Feature Importance (Permutation)
    print("Calculating Hybrid Permutation Feature Importance...")
    result = permutation_importance(best_hybrid, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    importances = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar', color='indigo')
    plt.title('Hybrid Ensemble Feature Importance (Permutation)')
    plt.tight_layout()
    plt.savefig('Hybrid_Feature_Importance.png')
    plt.close()

    with open('model_hybrid_tuned.pkl', 'wb') as f:
        pickle.dump(best_hybrid, f)
    print("Saved 'Hybrid_Confusion_Matrix.png', 'Hybrid_Feature_Importance.png', and 'model_hybrid_tuned.pkl'")

if __name__ == "__main__":
    train_tuned_hybrid()