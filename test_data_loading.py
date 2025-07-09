import pandas as pd

# Test data loading
try:
    df = pd.read_csv('EURUSD1.csv', header=None, names=['time', 'open', 'high', 'low', 'close', 'volume'], sep='\t', engine='python')
    print(f"Successfully loaded {len(df)} rows")
    print("First few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nAny NaN values:")
    print(df.isna().sum())
except Exception as e:
    print(f"Error: {e}")