import pandas as pd
import os

# --- Configuration ---
# You can change this to the feather file you want to inspect
FEATHER_FILE_PATH = os.path.join('data', 'feather', 'AAPL.feather')

# --- Verification Script ---
def read_and_verify_feather(filepath):
    """
    Reads a Feather file and prints its contents and schema for verification.
    """
    print(f"--- Verifying Feather File: {filepath} ---")

    # 1. Check if the file exists
    if not os.path.exists(filepath):
        print(f"\n[FAIL] File not found at '{filepath}'")
        print("Please ensure you have run the preprocessing script and that the symbol's CSV existed in the source data directory.")
        return

    try:
        # 2. Read the Feather file
        df = pd.read_feather(filepath)
        print("\n[SUCCESS] Successfully read the Feather file.")

        # 3. Verify the schema and data types
        print("\n[INFO] DataFrame Info (schema, data types, memory usage):")
        # Use StringIO to capture the output of df.info()
        from io import StringIO
        buffer = StringIO()
        df.info(buf=buffer)
        print(buffer.getvalue())

        # 4. Display the first few rows
        print("\n[INFO] First 5 rows of the data:")
        print(df.head())

        # 5. Check for expected columns from your preprocess_data.py script
        expected_columns = ['timestamp', 'date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        actual_columns = df.columns.tolist()

        print("\n[INFO] Verifying columns...")
        if all(col in actual_columns for col in expected_columns) and len(expected_columns) == len(actual_columns):
            print("[SUCCESS] All expected columns are present and in the correct order.")
        else:
            print("[FAIL] Column mismatch detected.")
            print(f"Expected: {expected_columns}")
            print(f"Found:    {actual_columns}")

    except Exception as e:
        print(f"\n[FAIL] An error occurred while reading or verifying the file: {e}")

if __name__ == "__main__":
    read_and_verify_feather(FEATHER_FILE_PATH)