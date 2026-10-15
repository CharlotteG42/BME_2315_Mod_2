'''1. Listing of all column headers in the dataset.'''

import pandas as pd

df = pd.read_csv("Module2_FSHD_Dataset.csv")

'''1. Brief background information about the dataset.'''
patient_count = df["subject_id"].nunique()
missing_by_column = df.isna().sum()

print("Dataset background:")
print(f"Number of patients: {patient_count}")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

print("\nColumn names:")
print(*df.columns, sep="\n")

print("\nMissing data points:")
columns_with_missing_data = missing_by_column[missing_by_column > 0]
if columns_with_missing_data.empty:
    print("No missing data.")
else:
    print(columns_with_missing_data.to_string())
print(f"Total missing data points: {missing_by_column.sum()}")

'''2. Listing of all unique values.'''
for col in df.columns:
    print(f"\nUnique values in '{col}':")
    print(*df[col].dropna().unique(), sep="\n")