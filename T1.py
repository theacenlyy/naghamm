import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 📊 إعداد المظهر العام للرسوم البيانية
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

# ==========================================
# 📥 1. تحميل البيانات وتصحيح الأسماء تلقائياً (💡 حل ذكي ونهائي لجميع الأخطاء)
# ==========================================
df = pd.read_csv('dataset.csv')

# تنظيف الفراغات المخفية حول أسماء الأعمدة إن وجدت
df.columns = df.columns.str.strip()

# خوارزمية ذكية لمسح الكلمات المفتاحية في ملفكِ وتوحيدها فوراً لتفادي أي ValueError
rename_rules = {}
for col in df.columns:
    col_clean = col.lower().replace('_', ' ').replace('-', ' ').strip()
    if 'passenger' in col_clean:
        rename_rules[col] = 'Passenger_Count'
    elif 'fare' in col_clean:
        rename_rules[col] = 'Fare Amount'
    elif 'car' in col_clean:
        rename_rules[col] = 'Car Condition'
    elif 'traffic' in col_clean:
        rename_rules[col] = 'Traffic Conditions'
    elif 'hour' in col_clean:
        rename_rules[col] = 'Hour'
    elif 'weather' in col_clean:
        rename_rules[col] = 'Weather'
    elif 'distance' in col_clean or 'dist' in col_clean:
        if 'jfk' in col_clean: rename_rules[col] = 'JFK Dist'
        elif 'ewr' in col_clean: rename_rules[col] = 'EWR_Dist'
        elif 'lga' in col_clean: rename_rules[col] = 'LGA Dist'
        elif 'sol' in col_clean: rename_rules[col] = 'SOL Dist'
        elif col_clean == 'distance': rename_rules[col] = 'Distance'

df = df.rename(columns=rename_rules)
print("✅ تم فحص أعمدة البيانات وتوحيدها بنجاح واجتياز كل مشاكل التسمية!")


# =========================================================================
#  Q1) Is higher Passenger count associated with higher price?
# Plot Choice: Boxplot
# =========================================================================
plt.figure(figsize=(10, 6))
sns.boxplot(x='Passenger_Count', y='Fare Amount', data=df, palette='muted')
plt.title('Q1: Fare Amount Distribution by Passenger Count', fontsize=12, fontweight='bold')
plt.xlabel('Number of Passengers')
plt.ylabel('Fare Amount ($)')
plt.show()


# =========================================================================
#  Q2) Does car condition influence the fare amount?
# Plot Choice: Side-by-Side (Boxplot + Bar Chart)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.boxplot(x='Car Condition', y='Fare Amount', data=df, ax=axes[0], palette='pastel')
axes[0].set_title('Fare Distribution vs Car Condition')
sns.barplot(x='Car Condition', y='Fare Amount', data=df, ax=axes[1], palette='pastel', ci=None)
axes[1].set_title('Average Fare vs Car Condition')
plt.suptitle('Q2: Impact of Car Condition on Uber Fares', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# =========================================================================
#  Q3) At what hour of the day are fares typically highest?
#  Plot Choice: Line Chart
# =========================================================================
hourly_fare = df.groupby('Hour')['Fare Amount'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x='Hour', y='Fare Amount', data=hourly_fare, marker='o', color='b', linewidth=2.5)
plt.title('Q3: Average Fare Amount by Hour of the Day (Trend Analysis)', fontsize=12, fontweight='bold')
plt.xlabel('Hour of the Day (0-23)')
plt.ylabel('Mean Fare Amount ($)')
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# =========================================================================
#  Q4) Does traffic condition affect the trip fare?
#  Plot Choice: Boxplot (هنا كان يحدث الخطأ وتم حله بالكامل!)
# =========================================================================
plt.figure(figsize=(10, 6))
sns.boxplot(x='Traffic Conditions', y='Fare Amount', data=df, palette='Set2')
plt.title('Q4: Impact of Traffic Conditions on Fare Amount', fontsize=12, fontweight='bold')
plt.xlabel('Traffic Condition')
plt.ylabel('Fare Amount ($)')
plt.show()


# =========================================================================
#  Q5) Is trip distance the strongest predictor of the fare amount?
#  Plot Choice: Scatter Plot + Correlation Heatmap
# =========================================================================
# الجزء الأول: Scatter Plot
plt.figure(figsize=(10, 5))
sns.scatterplot(x='Distance', y='Fare Amount', data=df, alpha=0.5, color='purple')
plt.title('Q5 (Part 1): Trip Distance vs Fare Amount', fontsize=12, fontweight='bold')
plt.xlabel('Total Trip Distance (km)')
plt.ylabel('Fare Amount ($)')
plt.show()

# الجزء الثاني: Heatmap
numeric_cols = ['Fare Amount', 'Distance', 'JFK Dist', 'EWR_Dist', 'LGA Dist', 'SOL Dist', 'Hour', 'Passenger_Count']
numeric_cols = [col for col in numeric_cols if col in df.columns]
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Q5 (Part 2): Correlation Heatmap (Identifying Strongest Predictor)', fontsize=12, fontweight='bold')
plt.show()


# =========================================================================
# Q6) At what time of day are ride requests most frequent? (Demand Analysis)
#  Plot Choice: Count Plot
# =========================================================================
plt.figure(figsize=(12, 6))
sns.countplot(x='Hour', data=df, palette='viridis')
plt.title('Q6: Frequency of Ride Requests by Hour of the Day (Demand Analysis)', fontsize=12, fontweight='bold')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Ride Requests (Count)')
plt.xticks(range(0, 24))
plt.show()


# =========================================================================
#  Q7) Does weather condition influence the average trip distance?
#  Plot Choice: Bar Chart
# =========================================================================
plt.figure(figsize=(8, 6))
sns.barplot(x='Weather', y='Distance', data=df, palette='magma', ci=None)
plt.title('Q7: Average Trip Distance by Weather Condition', fontsize=12, fontweight='bold')
plt.xlabel('Weather Condition')
plt.ylabel('Average Distance (km)')
plt.show()


# =========================================================================
#  Q8) Are rides that start closer to airports generally more expensive?
#  Plot Choice: Faceted Scatter Plots
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)


if 'JFK Dist' in df.columns:
    sns.scatterplot(x='JFK Dist', y='Fare Amount', data=df, alpha=0.4, color='teal', ax=axes[0])
    axes[0].set_title('JFK Airport Distance vs Fare')
    axes[0].set_xlabel('Distance to JFK (km)')
    axes[0].set_ylabel('Fare Amount ($)')

if 'EWR_Dist' in df.columns:
    sns.scatterplot(x='EWR_Dist', y='Fare Amount', data=df, alpha=0.4, color='coral', ax=axes[1])
    axes[1].set_title('EWR Airport Distance vs Fare')
    axes[1].set_xlabel('Distance to EWR (km)')


if 'LGA Dist' in df.columns:
    sns.scatterplot(x='LGA Dist', y='Fare Amount', data=df, alpha=0.4, color='gold', ax=axes[2])
    axes[2].set_title('LGA Airport Distance vs Fare')
    axes[2].set_xlabel('Distance to LGA (km)')

plt.suptitle('Q8: Analysis of Airport Proximity Impact on Uber Fares', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
