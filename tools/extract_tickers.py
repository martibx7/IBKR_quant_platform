import os
import re

def extract_tickers_from_line(line):
    """
    Extracts one or more comma-separated tickers from a line.
    The expected format is: "YYYY-MM-DD HH:MM:SS TICKER1, TICKER2, ..."
    """
    # Regex to capture all text that comes after the timestamp.
    match = re.match(r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\s(.+)", line.strip())

    if match:
        # This group contains the potential ticker string.
        tickers_string = match.group(1).strip()

        # Filter out known non-ticker log messages that start with a timestamp
        # These are specific patterns observed in your provided log files.
        if any(keyword in tickers_string for keyword in [
            "Launching analysis",
            "Accurate daily end-times",
            "Algorithm starting warm up",
            "Processing algorithm warm-up",
            "Algorithm finished warming up",
            "──────────────────────────────────────────────────────────────────────",
            "Ticker Logging Complete",
            "Total unique tickers logged",
            "Algorithm Id:"
        ]):
            return []

        # Split the string by commas and strip any extra whitespace from each potential ticker.
        potential_tickers = [ticker.strip() for ticker in tickers_string.split(',') if ticker.strip()]

        valid_tickers = []
        for ticker in potential_tickers:
            # Further validate each potential ticker:
            # - Must consist of uppercase letters, numbers, periods, or hyphens.
            # - Should not contain lowercase letters or other special characters (e.g., "%", "seconds").
            # - Length check (typically 1-10 characters for stock tickers).
            if re.fullmatch(r"^[A-Z0-9\.-]+$", ticker) and 1 <= len(ticker) <= 10:
                valid_tickers.append(ticker)

        return valid_tickers

    return [] # Return an empty list if the line doesn't match the expected timestamp format.

def find_and_process_log_files(output_filename="tickers.txt"):
    """
    Scans the current directory for .txt files, extracts all unique tickers,
    and writes them to the specified output file.
    """
    all_tickers = set()
    current_directory = os.getcwd()
    files_processed_count = 0

    print(f"Scanning for .txt files in: {current_directory}")

    # Get list of files in current directory to process
    files_to_process = [f for f in os.listdir(current_directory) if f.endswith(".txt") and f != output_filename]

    # Sort files by name for consistent processing order
    files_to_process.sort()

    for filename in files_to_process:
        print(f"Processing file: {filename}")
        files_processed_count += 1
        try:
            with open(os.path.join(current_directory, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    tickers_from_line = extract_tickers_from_line(line)
                    if tickers_from_line:
                        all_tickers.update(tickers_from_line)
        except Exception as e:
            print(f"  Error processing file {filename}: {e}")

    if not all_tickers:
        print("No tickers found in any of the processed .txt files.")
        return

    # Sort the unique tickers alphabetically before writing to the file.
    sorted_tickers = sorted(list(all_tickers))

    try:
        with open(os.path.join(current_directory, output_filename), 'w', encoding='utf-8') as outfile:
            for ticker in sorted_tickers:
                outfile.write(ticker + "\n")
        print("-" * 50)
        print(f"✅ Successfully extracted {len(sorted_tickers)} unique tickers.")
        print(f"Results written to: {os.path.join(current_directory, output_filename)}")
        print(f"Total .txt files processed: {files_processed_count}")
        print("-" * 50)
    except Exception as e:
        print(f"Error writing to the output file {output_filename}: {e}")

if __name__ == "__main__":
    find_and_process_log_files()