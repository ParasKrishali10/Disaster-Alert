import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# ==========================================
# 1. LOAD THE LABELED DATA
# ==========================================
filename = 'Labeled_Fire_Risk_Data.csv'
print(f"Loading heavily imbalanced data from {filename}...\n")
df = pd.read_csv(filename)

# Define features (X) and target (y)
# We drop the text label 'Risk_Category' because SMOTE only works with numbers
X = df.drop(columns=['Risk_Category', 'Risk_Level_Numeric'])
y = df['Risk_Level_Numeric']

print("--- ORIGINAL IMBALANCED DISTRIBUTION ---")
# 0: Low Risk, 1: Moderate Risk, 2: High Risk, 3: Extreme Risk
print(y.value_counts().sort_index())

# ==========================================
# 2. APPLY THE CORRECTED SMOTE SAMPLING
# ==========================================
print("\nApplying SMOTE to mathematically generate realistic disaster scenarios...")

# Find the exact size of your rarest class (which caused the error)
min_class_size = y.value_counts().min()

# SMOTE requires k_neighbors to be strictly less than the number of samples in the rarest class.
# If the rarest class has 2 samples, k_neighbors must be 1.
safe_k_neighbors = min_class_size - 1 

# Fallback failsafe: If a class somehow only has 1 sample, we force k_neighbors=1 
# (though ideally, you should always have at least 2 samples per class)
if safe_k_neighbors < 1:
    safe_k_neighbors = 1

print(f"Automatically adjusted SMOTE k_neighbors to: {safe_k_neighbors}")

# Initialize SMOTE with the corrected parameter
smote = SMOTE(random_state=42, k_neighbors=safe_k_neighbors)

# Generate the synthetic data
X_balanced, y_balanced = smote.fit_resample(X, y)

# ==========================================
# 3. REBUILD THE DATAFRAME
# ==========================================
# Put the generated features back into a clean Pandas DataFrame
balanced_df = pd.DataFrame(X_balanced, columns=X.columns)
balanced_df['Risk_Level_Numeric'] = y_balanced

# Map the numeric labels back to readable strings for your report charts
risk_mapping = {0: 'Low Risk', 1: 'Moderate Risk', 2: 'High Risk', 3: 'Extreme Risk'}
balanced_df['Risk_Category'] = balanced_df['Risk_Level_Numeric'].map(risk_mapping)

print("\n--- NEW BALANCED DISTRIBUTION ---")
print(balanced_df['Risk_Category'].value_counts())

# ==========================================
# 4. PLOT THE NEW REALISTIC DISTRIBUTION
# ==========================================
plt.figure(figsize=(10, 6))
sns.countplot(
    data=balanced_df, 
    x='Risk_Category', 
    order=['Low Risk', 'Moderate Risk', 'High Risk', 'Extreme Risk'],
    palette=['#2ca02c', '#ff7f0e', '#d62728', '#8c564b']
)
plt.title('Balanced Fire Risk Distribution (After SMOTE)', fontsize=15, fontweight='bold')
plt.ylabel('Number of Days (Including Synthetic)', fontsize=12)
plt.xlabel('Assessed Risk Level', fontsize=12)

# Add exact count labels to the top of the bars
for p in plt.gca().patches:
    plt.gca().annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 9), textcoords='offset points')

plt.tight_layout()
plt.savefig('Balanced_Risk_Distribution.png')
plt.show()

# ==========================================
# 5. EXPORT THE FINAL TRAINING DATA
# ==========================================
output_filename = 'Balanced_Fire_Risk_Data.csv'
balanced_df.to_csv(output_filename, index=False)
print(f"\nSuccess! Your perfectly balanced ML dataset is saved as '{output_filename}'.")