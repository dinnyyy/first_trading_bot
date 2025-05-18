import pandas as pd
from pathlib import Path

# 1. First verify the file exists
csv_path = Path('hmm/7_regimes/volar_frsi_ts_volz/bitcoin_state_changes.csv')
print(f"File exists: {csv_path.exists()}")

# 2. If it exists, inspect the columns
if csv_path.exists():
    # Read without parsing dates first
    temp_df = pd.read_csv(csv_path)
    print("Columns in CSV:", temp_df.columns.tolist())
    
    # If columns are correct, try reading with parse_dates
    try:
        state_df = pd.read_csv(csv_path, parse_dates=['Start Time', ' End Time'])
        print("Successfully loaded CSV with date parsing")
    except Exception as e:
        print(f"Error parsing dates: {e}")
else:
    print("File not found. Current working directory:", Path.cwd())