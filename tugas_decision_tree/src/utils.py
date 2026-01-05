"""
utils.py
Fungsi helper utility
"""

import pandas as pd
import numpy as np
import os

def print_dataset_info(df, name="Dataset"):
    """Print informasi dataset"""
    print(f"{'='*50}")
    print(f"{name.upper()} INFORMATION")
    print(f"{'='*50}")
    
    print(f"Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    
    print(f"\nData Types:")
    print(df.dtypes.value_counts())
    
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percent': missing_percent
    })
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    print(f"\nFirst 5 rows:")
    print(df.head())

def save_results(results_df, filename='results.csv'):
    """Save hasil experiment ke CSV"""
    results_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def create_directory(path):
    """Create directory jika belum ada"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")