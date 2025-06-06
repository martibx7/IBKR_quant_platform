import os
import re

def extract_tickers_from_line(line):
    """
    Extracts a ticker from a line if it matches the expected log format.
    Expected format: "YYYY-MM-DD HH:MM:SS TICKER"
    A ticker is assumed to be an all-uppercase string.
    """
    # Regex to capture date, time, and the potential ticker
    # It looks for:
    # ^                   - start of the line
    # \d{4}-\d{2}-\d{2}  - date like YYYY-MM-DD
    # \s                  - a space
    # \d{2}:\d{2}:\d{2}  - time like HH:MM:SS
    # \s                  - a space
    # ([A-Z0-9]+)         - the ticker (captured group 1): one or more uppercase letters/numbers
    # $                   - end of the line
    match = re.match(r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\s([A-Z0-9]+)$", line.strip())
    if match:
        return match.group(1)
    return None

def find_and_process_log_files(output_filename="tickers.txt"):
    """
    Scans the current directory for .txt files, extracts tickers,
    and writes unique tickers to the output file.
    """
    all_tickers = set()
    current_directory = os.getcwd()
    files_processed_count = 0
    tickers_found_count = 0

    print(f"Scanning for .txt files in: {current_directory}")

    for filename in os.listdir(current_directory):
        if filename.endswith(".txt") and filename != output_filename:
            print(f"Processing file: {filename}")
            files_processed_count += 1
            try:
                with open(os.path.join(current_directory, filename), 'r', encoding='utf-8') as f:
                    for line_number, line in enumerate(f, 1):
                        ticker = extract_tickers_from_line(line)
                        if ticker:
                            if ticker not in all_tickers:
                                # print(f"  Found new ticker: {ticker}") # Optional: for verbose output
                                all_tickers.add(ticker)
                                tickers_found_count += 1
                            # else:
                            # print(f"  Found duplicate ticker: {ticker}") # Optional
            except Exception as e:
                print(f"  Error processing file {filename}: {e}")

    if not all_tickers:
        print("No tickers found in any .txt files.")
        return

    # Sort tickers alphabetically before writing
    sorted_tickers = sorted(list(all_tickers))

    try:
        with open(os.path.join(current_directory, output_filename), 'w', encoding='utf-8') as outfile:
            for ticker in sorted_tickers:
                outfile.write(ticker + "\n")
        print(f"\nSuccessfully extracted {len(sorted_tickers)} unique tickers.")
        print(f"Results written to: {os.path.join(current_directory, output_filename)}")
        print(f"Total .txt files processed: {files_processed_count}")
    except Exception as e:
        print(f"Error writing to output file {output_filename}: {e}")

if __name__ == "__main__":
    find_and_process_log_files()
