"""
clean_transform.py

Data cleaning & transformation for ATX Traffic Incident Analysis.
Standardizes messy categories, handles nulls, removes duplicates.

Usage:
    python clean_transform.py \
        --input 'path/to/Real-Time_Traffic_Incident_Reports.csv' \
        --output 'data/processed/incidents_cleaned.csv' \
        --date_start '2023-01-01' \
        --date_end '2026-09-30'
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime


# ============================================================================
# INCIDENT TYPE STANDARDIZATION MAPPING
# ============================================================================

INCIDENT_TYPE_MAPPING = {
    # Crash-related
    'Crash Urgent': 'Crash Urgent',
    'Crash Service': 'Crash Service',
    'COLLISION': 'Collision',
    'COLLISION WITH INJURY': 'Collision with Injury',
    'COLLISION/PRIVATE PROPERTY': 'Collision Private Property',
    'COLLISN/ LVNG SCN': 'Collision Leaving Scene',
    'COLLISN / FTSRA': 'Collision',
    
    # Traffic hazards
    'Traffic Hazard': 'Traffic Hazard',
    'TRFC HAZD/ DEBRIS': 'Traffic Hazard',
    'Traffic Impediment': 'Traffic Impediment',
    'OBSTRUCT HWY': 'Obstruction',
    'BLOCKED DRIV/ HWY': 'Blocked Driveway',
    
    # Stalled vehicles
    'Stalled Vehicle': 'Stalled Vehicle',
    'zSTALLED VEHICLE': 'Stalled Vehicle',
    
    # Hazardous conditions
    'ICY ROADWAY': 'Icy Roadway',
    'HIGH WATER': 'High Water',
    'LOOSE LIVESTOCK': 'Loose Livestock',
    
    # Fleet-related
    'FLEET ACC/ INJURY': 'Fleet Accident with Injury',
    'FLEET ACC/ FATAL': 'Fleet Accident Fatal',
    
    # Severe incidents
    'TRAFFIC FATALITY': 'Traffic Fatality',
    'VEHICLE FIRE': 'Vehicle Fire',
    'AUTO/ PED': 'Auto Pedestrian',
    'BOAT ACCIDENT': 'Boat Accident',
    
    # Catch-all for unknown patterns
    'N / HZRD TRFC VIOL': 'Traffic Violation',
}


def clean_incident_types(df):
    """
    Standardize incident type categories.
    Maps all variants to consistent naming. Unmapped types stay original.
    """
    df['Issue Reported'] = df['Issue Reported'].str.strip()
    df['Issue Reported'] = df['Issue Reported'].map(
        lambda x: INCIDENT_TYPE_MAPPING.get(x, x)
    )
    return df


def clean_agency(df):
    """
    Standardize Agency field.
    Removes trailing whitespace, handles null values.
    """
    df['Agency'] = df['Agency'].fillna('Unknown')
    df['Agency'] = df['Agency'].str.strip()
    df['Agency'] = df['Agency'].replace('', 'Unknown')
    return df


def clean_coordinates(df):
    """
    Validate and clean latitude/longitude.
    Removes rows where either is 0 or NaN (invalid).
    Austin bounds: Lat 30.0-30.5, Lon -97.9 to -97.6
    """
    initial_count = len(df)
    
    # Remove where lat/long are 0 (missing)
    df = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)]
    
    # Remove where either is NaN
    df = df.dropna(subset=['Latitude', 'Longitude'])
    
    # Optional: Filter to Austin geographic bounds
    austin_bounds = (
        (df['Latitude'] >= 29.9) & (df['Latitude'] <= 30.6) &
        (df['Longitude'] >= -98.0) & (df['Longitude'] <= -97.5)
    )
    df = df[austin_bounds]
    
    removed = initial_count - len(df)
    print(f"Removed {removed} rows with invalid coordinates ({removed/initial_count*100:.1f}%)")
    
    return df


def parse_dates(df):
    """
    Parse Published Date and Status Date to datetime.
    Handles timezone info (UTC).
    """
    df['Published Date'] = pd.to_datetime(df['Published Date'], utc=True)
    df['Status Date'] = pd.to_datetime(df['Status Date'], utc=True)
    return df


def remove_duplicates(df):
    """
    Remove exact duplicates based on Traffic Report ID.
    """
    initial_count = len(df)
    df = df.drop_duplicates(subset=['Traffic Report ID'], keep='first')
    removed = initial_count - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate rows ({removed/initial_count*100:.2f}%)")
    return df


def filter_date_range(df, date_start, date_end):
    """
    Filter to specified date range.
    """
    mask = (df['Published Date'] >= date_start) & (df['Published Date'] <= date_end)
    removed = len(df) - mask.sum()
    df = df[mask]
    print(f"Filtered to date range {date_start} - {date_end}")
    print(f"  Removed {removed} rows outside range")
    return df


def clean_and_transform(input_path, output_path, date_start='2023-01-01', date_end='2026-09-30'):
    """
    Main cleaning pipeline.
    """
    print("=" * 80)
    print("ATX TRAFFIC INCIDENT DATA: CLEANING & TRANSFORMATION")
    print("=" * 80)
    
    # Load
    print(f"\n[1/8] Loading CSV: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Initial checks
    print(f"\n[2/8] Initial data quality checks")
    print(f"  Missing values:")
    print(f"    Latitude: {df['Latitude'].isna().sum():,}")
    print(f"    Longitude: {df['Longitude'].isna().sum():,}")
    print(f"    Agency: {df['Agency'].isna().sum():,}")
    print(f"    Status: {df['Status'].isna().sum():,}")
    
    # Parse dates
    print(f"\n[3/8] Parsing dates")
    df = parse_dates(df)
    print(f"  Date range in file: {df['Published Date'].min()} to {df['Published Date'].max()}")
    
    # Remove duplicates
    print(f"\n[4/8] Removing duplicates")
    df = remove_duplicates(df)
    print(f"  {len(df):,} rows remaining")
    
    # Filter date range
    date_start = pd.to_datetime(date_start, utc=True)
    date_end = pd.to_datetime(date_end, utc=True)
    print(f"\n[5/8] Filtering to date range")
    df = filter_date_range(df, date_start, date_end)
    print(f"  {len(df):,} rows remaining")
    
    # Clean coordinates
    print(f"\n[6/8] Validating & cleaning coordinates")
    df = clean_coordinates(df)
    print(f"  {len(df):,} rows remaining")
    
    # Standardize categories
    print(f"\n[7/8] Standardizing incident types & agencies")
    df = clean_incident_types(df)
    df = clean_agency(df)
    print(f"  Unique incident types: {df['Issue Reported'].nunique()}")
    print(f"  Unique agencies: {df['Agency'].nunique()}")
    
    # Final validation
    print(f"\n[8/8] Final validation")
    print(f"  Total rows after cleaning: {len(df):,}")
    print(f"  Date range: {df['Published Date'].min()} to {df['Published Date'].max()}")
    print(f"  Null values remaining:")
    print(f"    Latitude: {df['Latitude'].isna().sum()}")
    print(f"    Longitude: {df['Longitude'].isna().sum()}")
    print(f"    Issue Reported: {df['Issue Reported'].isna().sum()}")
    
    # Save
    print(f"\n[SAVE] Writing cleaned data to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(df):,} rows")
    
    print("\n" + "=" * 80)
    print("CLEANING COMPLETE")
    print("=" * 80)
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean ATX traffic incident data')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to raw CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output cleaned CSV'
    )
    parser.add_argument(
        '--date_start',
        type=str,
        default='2023-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--date_end',
        type=str,
        default='2026-09-30',
        help='End date (YYYY-MM-DD)'
    )
    
    args = parser.parse_args()
    clean_and_transform(args.input, args.output, args.date_start, args.date_end)
