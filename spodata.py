import pandas as pd
import glob
import os
import json
import re

# --- Configuration ---
# The script assumes it's in the same folder as your CSV files.
path_to_csv_folder = '.' 
output_filename = 'spotify_history_2016-2024_cleaned.json'

# --- Main Script ---
# Find all CSV files in the specified folder
all_files = glob.glob(os.path.join(path_to_csv_folder, "*.csv"))

all_records = []
print("Starting data processing...")

for file_path in all_files:
    filename = os.path.basename(file_path)
    print(f"Processing file: {filename}")

    # --- 1. Precise Year Extraction ---
    # Use a regular expression to find the 4-digit year in "your_top_songs_YYYY.csv"
    match = re.search(r'your_top_songs_(\d{4})\.csv', filename)
    
    if not match:
        print(f"  - WARNING: Skipping file '{filename}' as it does not match the expected format.")
        continue
        
    year = int(match.group(1))

    # --- 2. Load and Clean Dataframe ---
    df = pd.read_csv(file_path)
    df['Year'] = year # Add the correctly extracted year column

    # Convert date columns, turning any errors into 'Not a Time' (NaT)
    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
    df['Added At'] = pd.to_datetime(df['Added At'], errors='coerce')

    # Fill any missing 'Genres' with an empty string to prevent errors
    df['Genres'] = df['Genres'].fillna('')

    # Split the 'Genres' string into a clean list.
    # This also handles cases with empty strings, resulting in an empty list [].
    df['Genres'] = df['Genres'].apply(lambda x: [genre.strip() for genre in x.split(',') if genre.strip()])

    # --- 3. Prepare for JSON Conversion ---
    # Replace pandas-specific null values (NaN, NaT) with Python's None.
    # `None` will be converted to `null` in the JSON file, which is valid.
    df = df.where(pd.notnull(df), None)

    # Add the processed records from this file to our master list
    all_records.extend(df.to_dict('records'))

# --- 4. Export to a Clean JSON File ---
print(f"\nWriting {len(all_records)} total records to '{output_filename}'...")
with open(output_filename, 'w', encoding='utf-8') as f:
    # `indent=4` makes the file human-readable (optional).
    # `default=str` is a safeguard to handle any unexpected data types.
    json.dump(all_records, f, indent=4, default=str)

print("\nProcessing complete!")
print(f"The cleaned file '{output_filename}' is ready for MongoDB import.")
