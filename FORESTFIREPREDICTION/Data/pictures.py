import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. SETUP: DARK THEME & NEON COLORS
# ==========================================
# Match the reference image's dark, modern aesthetic
plt.style.use('dark_background')
fig_bg_color = '#121212' # Deep dark gray/black
axes_bg_color = '#121212'
text_color = '#ffffff'

# Neon palette based on your image (Cyan, Purple, Orange, Red)
neon_colors = ['#00E5FF', '#B388FF', '#FF9100', '#FF1744']
sns.set_palette(sns.color_palette(neon_colors))

# Load your data
df = pd.read_csv('Final_Forest_Fire_Data.csv')

# ==========================================
# CHART 1: RISK DISTRIBUTION (The Donut Chart)
# ==========================================
plt.figure(figsize=(6, 6), facecolor=fig_bg_color)
ax1 = plt.gca()
ax1.set_facecolor(axes_bg_color)

risk_counts = df['Risk_Category'].value_counts()
# Ensure consistent ordering
order = ['Low Risk', 'Moderate Risk', 'High Risk', 'Extreme Risk']
counts = [risk_counts.get(risk, 0) for risk in order]

plt.pie(counts, labels=order, colors=neon_colors, autopct='%1.1f%%', 
        startangle=90, textprops={'color': text_color, 'fontsize': 12, 'weight': 'bold'},
        wedgeprops=dict(width=0.4, edgecolor=fig_bg_color, linewidth=2)) # width creates the donut hole

plt.title('Overall Risk Assessment', color=text_color, fontsize=16, pad=20, weight='bold')
plt.savefig('Slide_Donut_Chart.png', dpi=300, bbox_inches='tight', facecolor=fig_bg_color)
plt.show()

# ==========================================
# CHART 2: FIRE DANGER TREND (The Line Chart)
# ==========================================
plt.figure(figsize=(10, 5), facecolor=fig_bg_color)
ax2 = plt.gca()
ax2.set_facecolor(axes_bg_color)

# Group by month to show a trend
monthly_trend = df.groupby('Month')['Fire_Danger_Index'].mean().reset_index()

plt.plot(monthly_trend['Month'], monthly_trend['Fire_Danger_Index'], 
         color='#00E5FF', linewidth=3, marker='o', markersize=8, 
         markeredgecolor='#ffffff', markeredgewidth=2)

# Fill under the line for that "dashboard" glowing effect
plt.fill_between(monthly_trend['Month'], monthly_trend['Fire_Danger_Index'], 
                 color='#00E5FF', alpha=0.1)

plt.title('Fire Danger Trend by Month', color=text_color, fontsize=16, pad=15, weight='bold')
plt.xlabel('Month', color='#aaaaaa', fontsize=12)
plt.ylabel('Avg Fire Danger Index', color='#aaaaaa', fontsize=12)
plt.grid(color='#333333', linestyle='--', linewidth=0.5, alpha=0.7)

# Remove top and right borders for a cleaner look
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_color('#555555')
ax2.spines['left'].set_color('#555555')

plt.savefig('Slide_Line_Chart.png', dpi=300, bbox_inches='tight', facecolor=fig_bg_color)
plt.show()

# ==========================================
# CHART 3: IMPACT FACTORS (The Bar Chart)
# ==========================================
plt.figure(figsize=(10, 5), facecolor=fig_bg_color)
ax3 = plt.gca()
ax3.set_facecolor(axes_bg_color)

# Show how 7-Day Temp changes across risk levels
temp_by_risk = df.groupby('Risk_Category')['Temp_7d_Avg'].mean().reindex(order)

sns.barplot(x=temp_by_risk.index, y=temp_by_risk.values, palette=neon_colors, ax=ax3)

plt.title('7-Day Avg Temperature vs Risk Level', color=text_color, fontsize=16, pad=15, weight='bold')
plt.xlabel('Risk Category', color='#aaaaaa', fontsize=12)
plt.ylabel('Temperature (°C)', color='#aaaaaa', fontsize=12)
plt.grid(color='#333333', linestyle='--', linewidth=0.5, axis='y', alpha=0.7)

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_color('#555555')
ax3.spines['left'].set_color('#555555')

plt.savefig('Slide_Bar_Chart.png', dpi=300, bbox_inches='tight', facecolor=fig_bg_color)
plt.show()

print("Dashboard charts generated successfully!")