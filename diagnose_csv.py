# diagnose_csv.py
import pandas as pd
import sys
import os # <--- This line was missing

def diagnose_file(filepath):
    """
    Tries to read a CSV file and prints detailed information about its
    structure, especially its columns.
    """
    print(f"--- Diagnosing File: {filepath} ---")

    if not os.path.exists(filepath):
        print(f"!!! ERROR: File not found at '{filepath}'. Please check the path. !!!")
        return

    try:
        # Read the file, assuming a header exists on the first line
        df = pd.read_csv(filepath)

        print("\n[1] Columns as read by pandas:")
        print(df.columns)
        print("\n[2] List of column names:")
        print(df.columns.tolist())
        print("\n[3] First 3 rows of data:")
        print(df.head(3))

    except Exception as e:
        print(f"\n!!! An error occurred while reading the file: {e} !!!")
        print("This might indicate a formatting issue with the CSV itself.")

if __name__ == "__main__":
    # Check if a file path was provided as a command-line argument
    if len(sys.argv) > 1:
        file_to_check = sys.argv[1]
    else:
        # Otherwise, use a default file for the check
        file_to_check = 'data/historical/AEE.csv'
        print(f"No file provided. Using default: {file_to_check}")

    diagnose_file(file_to_check)