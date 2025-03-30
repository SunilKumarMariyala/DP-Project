import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_excel('final_dataset.xlsx')

# Display basic information
print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nSample Data (First 3 rows):")
print(df.head(3))
print("\nMissing Values:")
print(df.isnull().sum())

# Check if there's a target column
if 'Fault_Type' not in df.columns:
    print("\nNote: No 'Fault_Type' column found in the dataset")
