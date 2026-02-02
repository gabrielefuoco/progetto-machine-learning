import pandas as pd
import os

file_path = 'data/fraud_dataset_v2.csv'
if not os.path.exists(file_path):
    print("Dataset not found!")
    exit()

print(f"Reading {file_path}...")
df = pd.read_csv(file_path)
print(f"Shape: {df.shape}")

categorical_attr_names = ['KTOSL', 'PRCTR', 'BSCHL', 'HKONT', 'BUKRS', 'WAERS']
print("\nCardinality of categorical columns:")
total_cols = 0
for col in categorical_attr_names:
    if col in df.columns:
        unique_count = df[col].nunique()
        print(f"  {col}: {unique_count}")
        total_cols += unique_count
    else:
        print(f"  {col}: NOT FOUND")

print(f"\nEstimated one-hot columns: {total_cols}")
print(f"Estimated boolean matrix size: {df.shape[0]} rows * {total_cols} cols * 1 byte = {df.shape[0] * total_cols / 1024**2:.2f} MB")
print(f"Estimated int64 matrix size: {df.shape[0]} rows * {total_cols} cols * 8 bytes = {df.shape[0] * total_cols * 8 / 1024**2:.2f} MB")
