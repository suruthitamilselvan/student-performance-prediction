import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('student_data.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nBasic stats:")
print(df.describe().round(2))

# ── Plot 1: Distribution of Final Score ──────────────────────
plt.figure(figsize=(8, 5))
sns.histplot(df['final_score'], bins=30, kde=True, color='steelblue')
plt.title('Distribution of Final Score')
plt.xlabel('Final Score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plot1_score_distribution.png')
plt.show()
print("Plot 1 saved!")

# ── Plot 2: Correlation Heatmap ───────────────────────────────
plt.figure(figsize=(9, 6))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('plot2_correlation_heatmap.png')
plt.show()
print("Plot 2 saved!")

# ── Plot 3: Study Hours vs Final Score ───────────────────────
plt.figure(figsize=(8, 5))
sns.scatterplot(x='study_hours', y='final_score', data=df,
                hue='pass_fail', palette={1:'green', 0:'red'}, alpha=0.6)
plt.title('Study Hours vs Final Score')
plt.xlabel('Study Hours')
plt.ylabel('Final Score')
plt.legend(title='Pass/Fail', labels=['Fail', 'Pass'])
plt.tight_layout()
plt.savefig('plot3_studyhours_vs_score.png')
plt.show()
print("Plot 3 saved!")

# ── Plot 4: Attendance vs Final Score ────────────────────────
plt.figure(figsize=(8, 5))
sns.scatterplot(x='attendance_pct', y='final_score', data=df,
                hue='pass_fail', palette={1:'green', 0:'red'}, alpha=0.6)
plt.title('Attendance vs Final Score')
plt.xlabel('Attendance (%)')
plt.ylabel('Final Score')
plt.tight_layout()
plt.savefig('plot4_attendance_vs_score.png')
plt.show()
print("Plot 4 saved!")

# ── Plot 5: Boxplot of all features ──────────────────────────
plt.figure(figsize=(12, 5))
features = ['study_hours', 'attendance_pct', 'previous_marks',
            'assignments_done', 'sleep_hours', 'internet_hours']
df[features].boxplot()
plt.title('Boxplot of All Features')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('plot5_boxplot.png')
plt.show()
print("Plot 5 saved!")

# ── Plot 6: Pass vs Fail count ───────────────────────────────
plt.figure(figsize=(6, 4))
colors = ['red', 'green']
sns.countplot(x='pass_fail', data=df, palette=colors)
plt.title('Pass vs Fail Count')
plt.xticks([0, 1], ['Fail', 'Pass'])
plt.xlabel('Result')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plot6_pass_fail_count.png')
plt.show()
print("Plot 6 saved!")

print("\nAll 6 plots saved in your folder!")