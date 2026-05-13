import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 1. LOAD DATASET
# ==========================================
filename = 'Final_Forest_Fire_Data.csv'
print(f"Loading balanced dataset from {filename}...\n")
df = pd.read_csv(filename)

# Basic Dataset Info
print(f"Total Records: {df.shape[0]:,}")
print(f"Total Features: {df.shape[1]}")
print("\n--- Distribution of Target Variable ---")
print(df['Risk_Category'].value_counts())

# Set global plotting style for professional look
sns.set_theme(style="whitegrid")
custom_palette = ['#2ca02c', '#ff7f0e', '#d62728', '#8c564b']
risk_order = ['Low Risk', 'Moderate Risk', 'High Risk', 'Extreme Risk']

# ==========================================
# 2. VERIFY CLASS BALANCE (Bar Chart)
# ==========================================
plt.figure(figsize=(9, 5))
ax = sns.countplot(data=df, x='Risk_Category', order=risk_order, palette=custom_palette)
plt.title('Perfectly Balanced Risk Categories (Post-SMOTE)', fontsize=14, fontweight='bold')
plt.xlabel('Assessed Risk Level', fontsize=12)
plt.ylabel('Number of Days', fontsize=12)

# Add exact numbers above bars
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 8), textcoords='offset points')
plt.tight_layout()
plt.savefig('1_Balanced_Target_Distribution.png', dpi=300)
plt.show()

# ==========================================
# 3. CORRELATION HEATMAP
# ==========================================
# Select numerical columns
numeric_cols = df.select_dtypes(include=[np.number])
corr_matrix = numeric_cols.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Fire Features (Balanced Data)', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('2_Balanced_Correlation_Matrix.png', dpi=300)
plt.show()

print("\n--- Correlation with Risk Level ---")
print(corr_matrix['Risk_Level_Numeric'].sort_values(ascending=False).drop('Risk_Level_Numeric'))

# ==========================================
# 4. KEY DRIVERS ANALYSIS (Boxplots)
# ==========================================
# This proves that our synthetic data still obeys real-world weather patterns
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('How Weather Changes with Fire Risk Levels', fontsize=18, fontweight='bold')

# Plot 1: 7-Day Rainfall (The Suppressant)
sns.boxplot(ax=axes[0], data=df, x='Risk_Category', y='Rain_7d_Sum', order=risk_order, palette=custom_palette)
axes[0].set_title('Impact of 7-Day Rainfall', fontsize=14)
axes[0].set_ylabel('Total Rain (mm)')
axes[0].set_xlabel('')

# Plot 2: Max Temperature (The Fuel Baker)
sns.boxplot(ax=axes[1], data=df, x='Risk_Category', y='Max_Temperature_C', order=risk_order, palette=custom_palette)
axes[1].set_title('Impact of Max Temperature', fontsize=14)
axes[1].set_ylabel('Temperature (°C)')
axes[1].set_xlabel('Risk Category')

# Plot 3: Fire Danger Index (The Custom Proxy)
sns.boxplot(ax=axes[2], data=df, x='Risk_Category', y='Fire_Danger_Index', order=risk_order, palette=custom_palette)
axes[2].set_title('Fire Danger Index Distribution', fontsize=14)
axes[2].set_ylabel('Index Score')
axes[2].set_xlabel('')

plt.tight_layout()
plt.savefig('3_Key_Drivers_Boxplots.png', dpi=300)
plt.show()