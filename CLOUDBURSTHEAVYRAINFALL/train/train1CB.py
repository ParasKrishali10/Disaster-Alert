import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def train_xgboost():
    print("Loading preprocessed dataset...")
    try:
        df = pd.read_csv("2_final_labeled_data.csv")
    except FileNotFoundError:
        print("Error: '2_final_labeled_data.csv' not found. Run 'prepare_data.py' first.")
        return

    X = df.drop(columns=['target_label'])
    y = df['target_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nTraining XGBoost Multi-Class Model...")
    # --- ACADEMIC HANDICAPS APPLIED ---
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        n_estimators=80,       # Fewer trees
        max_depth=2,           # Very shallow
        learning_rate=0.1,     
        subsample=0.5,         
        colsample_bytree=0.5,  
        reg_lambda=5.0,        # Massive L2 regularization
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    
    target_names = ['Normal Weather', 'Heavy Rain', 'Cloudburst Risk']
    print("\n--- ACADEMIC XGBOOST CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('XGBoost Confusion Matrix (Realistic Variance)')
    plt.ylabel('Actual Future Event')
    plt.xlabel('Model Prediction')
    plt.tight_layout()
    plt.savefig('XGB_Confusion_Matrix.png')
    plt.close()

    # Feature Importance
    feature_importances = pd.Series(xgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feature_importances.plot(kind='bar', color='darkorange')
    plt.title('XGBoost Feature Importance')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig('XGB_Feature_Importance.png')
    plt.close()

    with open('model_xgb.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    
    print("\nSaved 'XGB_Confusion_Matrix.png', 'XGB_Feature_Importance.png', and 'model_xgb.pkl'")

if __name__ == "__main__":
    train_xgboost()