"""
Analyze churn_noisy.xlsx and assign meaningful column names.

The dataset has 17 columns with no proper headers (first data row was used as header).
This script:
  1. Reloads the data correctly (treating row 0 as data, not header)
  2. Assigns descriptive column names based on data analysis
  3. Produces a full EDA summary
  4. Saves the cleaned dataset as churn_noisy_named.xlsx
"""

import pandas as pd
import numpy as np

# -- Load dataset (NO header - the file has no column names) --
df = pd.read_excel('churn_noisy.xlsx', header=None)

print(f"Shape after loading with header=None: {df.shape}")
print(f"First 3 rows:\n{df.head(3)}\n")

# -- Column analysis --
# 17 columns total:
#   Cols 0-12  : float64 - numerical features (most are 0-1 normalized)
#   Col 13     : int/float - binary (0/1) -> target variable (Churn)
#   Col 14     : object - Age groups (Young, Adult, Middle, Senior, Elderly)
#   Col 15     : object - Plan type (Pay-as-you-go / Contractual)
#   Col 16     : object - Customer status (Active / Inactive)

column_names = [
    'AccountAge',             # 0  : float 0-1, normalized tenure/account age
    'DataUsage',              # 1  : float 0-1, normalized data usage
    'Age',                    # 2  : float ~3-47, actual age (with noise)
    'MonthlySpend',           # 3  : float -0.15 to 10.1, monthly spend (noisy)
    'CallDuration',           # 4  : float 0-1, normalized call duration
    'CallFrequency',          # 5  : float 0-1, normalized call frequency
    'SMSFrequency',           # 6  : float 0-1, normalized SMS frequency
    'DataFrequency',          # 7  : float 0-1, normalized data frequency
    'PlanDuration',           # 8  : float ~0.86-5.1, plan duration (noisy)
    'OverageFee',             # 9  : float ~0.85-2.14, overage fee (noisy)
    'RoamingCalls',           # 10 : float ~0.82-2.16, roaming calls (noisy)
    'CustomerSatisfaction',   # 11 : float 0-1, satisfaction score
    'ComplaintRate',          # 12 : float 0-1, normalized complaint rate
    'Churn',                  # 13 : int 0/1 - binary target
    'AgeGroup',               # 14 : categorical age group
    'PlanType',               # 15 : Pay-as-you-go / Contractual
    'Status',                 # 16 : Active / Inactive
]

df.columns = column_names

# -- Display full analysis --
print("=" * 70)
print("  CHURN NOISY DATASET - ANALYSIS REPORT")
print("=" * 70)

print(f"\n>> Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

# Column overview
print("-" * 70)
print("  COLUMN OVERVIEW")
print("-" * 70)
for i, col in enumerate(df.columns):
    dtype = df[col].dtype
    nulls = df[col].isnull().sum()
    if dtype in ['float64', 'int64']:
        print(f"  [{i:>2}] {col:<25} {str(dtype):<10}  "
              f"min={df[col].min():>8.4f}  max={df[col].max():>8.4f}  "
              f"mean={df[col].mean():>8.4f}  nulls={nulls}")
    else:
        unique = df[col].nunique()
        print(f"  [{i:>2}] {col:<25} {str(dtype):<10}  "
              f"unique={unique:<4}  values={df[col].unique().tolist()}  nulls={nulls}")

# Target distribution
print("\n" + "-" * 70)
print("  TARGET VARIABLE: Churn")
print("-" * 70)
churn_counts = df['Churn'].value_counts()
print(f"  0 (Not Churned): {churn_counts.get(0, 0):>5}  "
      f"({churn_counts.get(0, 0) / len(df) * 100:.1f}%)")
print(f"  1 (Churned):     {churn_counts.get(1, 0):>5}  "
      f"({churn_counts.get(1, 0) / len(df) * 100:.1f}%)")

# Categorical distributions
print("\n" + "-" * 70)
print("  CATEGORICAL DISTRIBUTIONS")
print("-" * 70)
for col in ['AgeGroup', 'PlanType', 'Status']:
    print(f"\n  {col}:")
    for val, count in df[col].value_counts().items():
        print(f"    {val:<20} {count:>5}  ({count / len(df) * 100:.1f}%)")

# Noise detection
print("\n" + "-" * 70)
print("  NOISE DETECTION")
print("-" * 70)
# Check for negative values in columns that should be non-negative
for col in ['MonthlySpend', 'PlanDuration', 'OverageFee', 'RoamingCalls']:
    neg_count = (df[col] < 0).sum()
    if neg_count > 0:
        print(f"  [!] {col}: {neg_count} negative values detected (likely noise)")

# Check for values outside expected ranges
for col in ['AccountAge', 'DataUsage', 'CallDuration', 'CallFrequency',
            'SMSFrequency', 'DataFrequency', 'CustomerSatisfaction', 'ComplaintRate']:
    out_range = ((df[col] < 0) | (df[col] > 1)).sum()
    if out_range > 0:
        print(f"  [!] {col}: {out_range} values outside [0,1] range (likely noise)")

no_noise = True
for col in ['MonthlySpend', 'PlanDuration', 'OverageFee', 'RoamingCalls']:
    if (df[col] < 0).sum() > 0:
        no_noise = False
        break
if no_noise:
    for col in ['AccountAge', 'DataUsage', 'CallDuration', 'CallFrequency',
                'SMSFrequency', 'DataFrequency', 'CustomerSatisfaction', 'ComplaintRate']:
        if ((df[col] < 0) | (df[col] > 1)).sum() > 0:
            no_noise = False
            break
if no_noise:
    print("  No obvious noise patterns detected in numerical ranges.")

# Correlation with churn
print("\n" + "-" * 70)
print("  CORRELATION WITH CHURN (Numerical Features)")
print("-" * 70)
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('Churn')
correlations = df[num_cols].corrwith(df['Churn']).sort_values(ascending=False)
for col, corr in correlations.items():
    bar = "#" * int(abs(corr) * 30)
    sign = "+" if corr >= 0 else "-"
    print(f"  {sign} {col:<25} {corr:>+.4f}  {bar}")

# Statistical summary
print("\n" + "-" * 70)
print("  STATISTICAL SUMMARY")
print("-" * 70)
print(df.describe().round(4).to_string())

# -- Save cleaned dataset --
output_file = 'churn_noisy_named.xlsx'
df.to_excel(output_file, index=False)
print(f"\n[OK] Cleaned dataset saved to: {output_file}")
print(f"     Columns: {df.columns.tolist()}")
