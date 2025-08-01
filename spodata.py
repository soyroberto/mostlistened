import pandas as pd
import glob
import os
import json
import re

# --- Main Script ---
path_to_csv_folder = '.'
output_filename = 'spotify_history_2016-2024_cleaned.json'

all_files = glob.glob(os.path.join(path_to_csv_folder, "*.csv"))
all_records = []
print("Starting data processing...")

for file_path in all_files:
    filename = os.path.basename(file_path)
    print(f"Processing file: {filename}")

    match = re.search(r'your_top_songs_(\d{4})\.csv', filename)
    if not match:
        print(f"  - WARNING: Skipping file '{filename}' as it does not match the expected format.")
        continue
    year = int(match.group(1))

    # --- 1. Load and Clean Dataframe using Pandas ---
    df = pd.read_csv(file_path)
    df['Year'] = year

    # Convert date columns, turning any errors into 'Not a Time' (NaT)
    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
    df['Added At'] = pd.to_datetime(df['Added At'], errors='coerce')

    # Fill any missing 'Genres' with an empty string to prevent errors
    df['Genres'] = df['Genres'].fillna('')

    # Split the 'Genres' string into a clean list of strings
    df['Genres'] = df['Genres'].apply(lambda x: [genre.strip() for genre in x.split(',') if genre.strip()])

    # --- 2. Convert DataFrame to JSON string with proper NaN handling ---
    # This is the key step. `to_json` has built-in options to handle this correctly.
    # `orient='records'` creates the list of dictionaries structure.
    # `default_handler=str` helps with datetime objects.
    # `double_precision=15` maintains numeric accuracy.
    json_string = df.to_json(
        orient='records', 
        default_handler=str, 
        double_precision=15
    )

    # --- 3. Load the JSON string back into a Python object ---
    # This step effectively converts all pandas-specific types (like NaT)
    # and non-standard values into standard Python/JSON types (like null).
    python_objects = json.loads(json_string)
    
    all_records.extend(python_objects)

# --- 4. Export the final, clean Python object to a file ---
print(f"\nWriting {len(all_records)} total records to '{output_filename}'...")
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(all_records, f, indent=4)

print("\nProcessing complete!")
print(f"The final cleaned file '{output_filename}' is ready for import.")
