#!/usr/bin/env python3
"""
CSV to JSON Converter for Music AI Dashboard
Converts CSV music data to the JSON format expected by the dashboard
"""

import pandas as pd
import json
import numpy as np
from datetime import datetime
import re

def convert_csv_to_json(csv_file_path, output_json_path=None):
    """
    Convert CSV music data to JSON format compatible with the Music AI Dashboard
    
    Args:
        csv_file_path: Path to the CSV file
        output_json_path: Optional path to save JSON file. If None, returns JSON data
    
    Returns:
        List of dictionaries in JSON format
    """
    
    # Read CSV file
    df = pd.read_csv(csv_file_path)
    
    # Clean and convert data
    converted_data = []
    
    for _, row in df.iterrows():
        # Create a new record
        record = {}
        
        # Copy basic string fields
        string_fields = ['Track URI', 'Track Name', 'Album Name', 'Artist Name(s)', 'Record Label']
        for field in string_fields:
            if field in row:
                record[field] = str(row[field]) if pd.notna(row[field]) else ""
        
        # Handle Release Date - convert to timestamp
        if 'Release Date' in row and pd.notna(row['Release Date']):
            try:
                # Try to parse the date
                if isinstance(row['Release Date'], str):
                    # Handle different date formats
                    date_str = str(row['Release Date'])
                    if len(date_str) == 4:  # Just year
                        date_obj = datetime(int(date_str), 1, 1)
                    else:
                        date_obj = pd.to_datetime(date_str)
                else:
                    date_obj = pd.to_datetime(row['Release Date'])
                
                # Convert to timestamp (milliseconds)
                record['Release Date'] = int(date_obj.timestamp() * 1000)
            except:
                record['Release Date'] = 0
        else:
            record['Release Date'] = 0
        
        # Handle Added At - convert to timestamp
        if 'Added At' in row and pd.notna(row['Added At']):
            try:
                added_at = pd.to_datetime(row['Added At'])
                record['Added At'] = int(added_at.timestamp() * 1000)
            except:
                record['Added At'] = int(datetime.now().timestamp() * 1000)
        else:
            record['Added At'] = int(datetime.now().timestamp() * 1000)
        
        # Extract Year from Added At for dashboard compatibility
        try:
            year = datetime.fromtimestamp(record['Added At'] / 1000).year
            record['Year'] = year
        except:
            record['Year'] = datetime.now().year
        
        # Handle numeric fields
        numeric_fields = [
            'Duration (ms)', 'Popularity', 'Key', 'Loudness', 'Mode', 'Time Signature',
            'Danceability', 'Energy', 'Speechiness', 'Acousticness', 
            'Instrumentalness', 'Liveness', 'Valence', 'Tempo'
        ]
        
        for field in numeric_fields:
            if field in row:
                try:
                    value = row[field]
                    if pd.notna(value):
                        # Convert to appropriate numeric type
                        if field in ['Duration (ms)', 'Popularity', 'Key', 'Mode', 'Time Signature']:
                            record[field] = int(float(value))
                        else:
                            record[field] = float(value)
                    else:
                        record[field] = 0 if field in ['Duration (ms)', 'Popularity', 'Key', 'Mode', 'Time Signature'] else 0.0
                except:
                    record[field] = 0 if field in ['Duration (ms)', 'Popularity', 'Key', 'Mode', 'Time Signature'] else 0.0
        
        # Handle boolean fields
        boolean_fields = ['Explicit']
        for field in boolean_fields:
            if field in row:
                try:
                    value = str(row[field]).lower()
                    record[field] = value in ['true', '1', 'yes', 'y']
                except:
                    record[field] = False
        
        # Handle Added By
        if 'Added By' in row:
            record['Added By'] = str(row['Added By']) if pd.notna(row['Added By']) else None
        
        # Handle Genres - convert comma-separated string to list
        if 'Genres' in row and pd.notna(row['Genres']):
            genres_str = str(row['Genres']).strip()
            if genres_str and genres_str != '':
                # Split by comma and clean up
                genres_list = [genre.strip() for genre in genres_str.split(',') if genre.strip()]
                record['Genres'] = genres_list
            else:
                record['Genres'] = []
        else:
            record['Genres'] = []
        
        converted_data.append(record)
    
    # Save to file if path provided
    if output_json_path:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    return converted_data

def validate_converted_data(data):
    """
    Validate that the converted data has the required fields for the dashboard
    
    Args:
        data: List of dictionaries (converted JSON data)
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    
    if not data:
        return False, ["No data found"]
    
    required_fields = [
        'Track Name', 'Artist Name(s)', 'Year', 'Danceability', 'Energy', 
        'Valence', 'Acousticness', 'Instrumentalness', 'Liveness', 'Speechiness'
    ]
    
    errors = []
    
    # Check first record for required fields
    first_record = data[0]
    missing_fields = [field for field in required_fields if field not in first_record]
    
    if missing_fields:
        errors.append(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Check data types
    sample_size = min(10, len(data))
    for i in range(sample_size):
        record = data[i]
        
        # Check numeric fields
        numeric_fields = ['Danceability', 'Energy', 'Valence', 'Acousticness', 
                         'Instrumentalness', 'Liveness', 'Speechiness']
        
        for field in numeric_fields:
            if field in record:
                try:
                    float(record[field])
                except:
                    errors.append(f"Record {i}: {field} is not numeric")
        
        # Check Year
        if 'Year' in record:
            try:
                year = int(record['Year'])
                if year < 1900 or year > 2030:
                    errors.append(f"Record {i}: Year {year} seems invalid")
            except:
                errors.append(f"Record {i}: Year is not a valid integer")
    
    return len(errors) == 0, errors

def test_conversion():
    """Test the conversion with the uploaded files"""
    try:
        # Test with the uploaded CSV
        csv_path = "/home/ubuntu/upload/soundhound.csv"
        converted_data = convert_csv_to_json(csv_path)
        
        # Validate
        is_valid, errors = validate_converted_data(converted_data)
        
        print(f"✅ Conversion successful!")
        print(f"📊 Converted {len(converted_data)} records")
        print(f"✅ Validation: {'Passed' if is_valid else 'Failed'}")
        
        if errors:
            print("⚠️ Validation errors:")
            for error in errors:
                print(f"  - {error}")
        
        # Show sample record
        if converted_data:
            print("\n📋 Sample converted record:")
            sample = converted_data[0]
            for key, value in sample.items():
                print(f"  {key}: {value} ({type(value).__name__})")
        
        return converted_data, is_valid, errors
        
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        return None, False, [str(e)]

if __name__ == "__main__":
    test_conversion()

