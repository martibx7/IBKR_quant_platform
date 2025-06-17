# process_profiler.py

import pstats
import io

# --- Configuration ---
# The filename of your profiler output
profiler_file = 'backtest.prof'
# The name of the text file we will create
output_file = 'profiler_results.txt'

# --- Script ---
try:
    # Create a string stream to capture the output text
    s = io.StringIO()

    # Load the stats from the .prof file, directing its output to our stream
    ps = pstats.Stats(profiler_file, stream=s)

    # Sort the statistics by cumulative time and print the top 50 functions
    # 'cumulative' is usually the most useful metric to find bottlenecks
    ps.sort_stats('cumulative').print_stats(50)

    # Save the captured text output to the final text file
    with open(output_file, 'w') as f:
        f.write(s.getvalue())

    print(f"Success! Profiler results have been saved to '{output_file}'")
    print("Please open that file, copy its entire contents, and paste it back into our chat.")

except FileNotFoundError:
    print(f"Error: The file '{profiler_file}' was not found.")
    print("Please make sure you have run the profiler first and that this script is in the same directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")