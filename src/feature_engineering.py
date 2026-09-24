"""
feature_engineering.py

Extract temporal features and create target variable for modeling.

Usage:
    python feature_engineering.py \
        --input 'data/processed/incidents_cleaned.csv' \
        --output 'data/processed/incidents_features.csv'
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


# ============================================================================
# INCIDENT SEVERITY CLASSIFICATION
# ============================================================================

URGENT_INCIDENTS = {
    'Crash Urgent',
    'Crash Service',
    'Collision',
    'Collision with Injury',
    'Collision Leaving Scene',
    'Traffic Fatality',
    'Auto Pedestrian',
    'Fleet Accident with Injury',
    'Fleet Accident Fatal',
    'Vehicle Fire',
}


def extract_temporal_features(df):
    """
    Extract hour, day of week, month, year, season, etc. from Published Date.
    """
    df['published_hour'] = df['Published Date'].dt.hour
    df['published_day_of_week'] = df['Published Date'].dt.dayofweek  # 0=Mon, 6=Sun
    df['published_day_name'] = df['Published Date'].dt.day_name()
    df['published_month'] = df['Published Date'].dt.month
    df['published_month_name'] = df['Published Date'].dt.month_name()
    df['published_year'] = df['Published Date'].dt.year
    df['published_quarter'] = df['Published Date'].dt.quarter
    
    # Week of year
    df['published_week'] = df['Published Date'].dt.isocalendar().week
    
    # Days since start (for trend analysis)
    min_date = df['Published Date'].min()
    df['days_since_start'] = (df['Published Date'] - min_date).dt.days
    
    return df


def create_categorical_features(df):
    """
    Create categorical features from temporal data.
    """
    # Is weekend
    df['is_weekend'] = df['published_day_of_week'].isin([5, 6]).astype(int)
    
    # Is weekday
    df['is_weekday'] = (~df['published_day_of_week'].isin([5, 6])).astype(int)
    
    # Rush hour (7-9 AM, 4-7 PM)
    df['is_rush_hour'] = df['published_hour'].isin([7, 8, 16, 17, 18]).astype(int)
    
    # Peak hours (noon-1pm, 5-6pm)
    df['is_peak_hour'] = df['published_hour'].isin([12, 17, 18]).astype(int)
    
    # Night time (10 PM - 5 AM)
    df['is_night'] = df['published_hour'].isin(range(22, 24)) | df['published_hour'].isin(range(0, 6))
    df['is_night'] = df['is_night'].astype(int)
    
    # Season (Northern Hemisphere)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['season'] = df['published_month'].apply(get_season)
    
    return df


def create_target_variable(df):
    """
    Create binary target: is_urgent
    1 = Urgent incident (Crash Urgent, Crash Service, etc.)
    0 = Non-urgent
    """
    df['is_urgent'] = df['Issue Reported'].isin(URGENT_INCIDENTS).astype(int)
    
    # Also create severity category for additional analysis
    def categorize_severity(incident_type):
        if incident_type in URGENT_INCIDENTS:
            return 'Urgent'
        else:
            return 'Non-Urgent'
    
    df['severity_category'] = df['Issue Reported'].apply(categorize_severity)
    
    return df


def calculate_response_time(df):
    """
    Calculate response time: time between incident report and status update.
    Returns minutes.
    """
    df['response_time_minutes'] = (
        (df['Status Date'] - df['Published Date']).dt.total_seconds() / 60
    ).round(2)
    
    # Remove negative response times (data errors)
    df.loc[df['response_time_minutes'] < 0, 'response_time_minutes'] = np.nan
    
    return df


def engineer_features(input_path, output_path):
    """
    Main feature engineering pipeline.
    """
    print("=" * 80)
    print("FEATURE ENGINEERING FOR ATX TRAFFIC INCIDENTS")
    print("=" * 80)
    
    # Load cleaned data
    print(f"\n[1/5] Loading cleaned data: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    df['Published Date'] = pd.to_datetime(df['Published Date'], utc=True)
    df['Status Date'] = pd.to_datetime(df['Status Date'], utc=True)
    print(f"  Loaded {len(df):,} rows")
    
    # Extract temporal features
    print(f"\n[2/5] Extracting temporal features")
    df = extract_temporal_features(df)
    print(f"  Added: hour, day_of_week, month, year, quarter, season, week, days_since_start")
    
    # Create categorical features
    print(f"\n[3/5] Creating categorical features")
    df = create_categorical_features(df)
    print(f"  Added: is_weekend, is_weekday, is_rush_hour, is_peak_hour, is_night, season")
    
    # Create target variable
    print(f"\n[4/5] Creating target variable (is_urgent)")
    df = create_target_variable(df)
    urgent_count = df['is_urgent'].sum()
    non_urgent_count = len(df) - urgent_count
    print(f"  Urgent incidents: {urgent_count:,} ({urgent_count/len(df)*100:.1f}%)")
    print(f"  Non-urgent incidents: {non_urgent_count:,} ({non_urgent_count/len(df)*100:.1f}%)")
    
    # Calculate response time
    print(f"\n[5/5] Calculating response time")
    df = calculate_response_time(df)
    print(f"  Response time range: {df['response_time_minutes'].min():.1f} to {df['response_time_minutes'].max():.1f} minutes")
    print(f"  Missing response times: {df['response_time_minutes'].isna().sum():,}")
    
    # Save
    print(f"\n[SAVE] Writing engineered data to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(df):,} rows with {len(df.columns)} columns")
    
    # Summary
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 80)
    print(f"\nNew columns created:")
    temporal_cols = [
        'published_hour', 'published_day_of_week', 'published_day_name',
        'published_month', 'published_month_name', 'published_year', 'published_quarter',
        'published_week', 'days_since_start'
    ]
    categorical_cols = [
        'is_weekend', 'is_weekday', 'is_rush_hour', 'is_peak_hour', 'is_night', 'season'
    ]
    target_cols = ['is_urgent', 'severity_category', 'response_time_minutes']
    
    print(f"\nTemporal ({len(temporal_cols)}): {', '.join(temporal_cols)}")
    print(f"\nCategorical ({len(categorical_cols)}): {', '.join(categorical_cols)}")
    print(f"\nTarget ({len(target_cols)}): {', '.join(target_cols)}")
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Engineer features for ATX traffic analysis')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to cleaned CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output features CSV'
    )
    
    args = parser.parse_args()
    engineer_features(args.input, args.output)
