'''
temporal_analysis.py

Phase A: Temporal Analysis of Austin Traffic Incidents
Focus: Trends over time, hourly/daily seasonality & statistical tests'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore') #supress pandas plotting warnings

def load_data(filepath):
    """ LOAD THE ENGINEERED FEATURE DATASET"""
    print("LOADING DATA...")
    df = pd.read_csv(filepath)
    #ENSURE PUBLISHED DATE IS A PROPER DATETIME OBJECT
    df['Published Date'] = pd.to_datetime(df['Published Date'])
    return df

def analyze_daily_trend(df, output_dir):
    """PLOT DAILY INCIDENT COUNTS WITH A 7-DAY ROLLING AVERAGE"""
    print('Generating Daily Trend Plot...')

    #COUNT INCIDENTS PER DAY
    daily_counts = df.groupby(df['Published Date'].dt.date).size()
    plt.figure(figsize=(14,6))

    #PLOT RAW DAILY COUNTS IN LIGHT BLUE
    plt.plot(daily_counts.index, daily_counts.values, color='skyblue', alpha=0.5, label='Daily Incidents')

    
     #PLOT 7DAY ROLLING AVERAGE IN DARK BLUE
    rolling_7d = daily_counts.rolling(window=7).mean() 
    plt.plot(rolling_7d.index, rolling_7d.values, color='darkblue', linewidth=2, label='7-Day Rolling Avg')
    
    plt.title('ATX Traffic Incidents: Daily Trend (2023-2026)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Incidents', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_dir / '1_daily_trend.png', dpi=300)
    plt.close()
    #this plot allows us to visualize the overall trend of incidents over 2023 -2026

def analyze_hourly_heatmap(df, output_dir):
    """Create a heatmap of incidents by Hour and Day of Week."""
    print("Generating Hourly Heatmap...")
    
    # Create a pivot table: Rows = Hour, Columns = Day of Week
    pivot = pd.crosstab(index=df['published_hour'], columns=df['published_day_name'])
    
    # Reorder columns to standard Mon-Sun
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot[days_order]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, cmap='YlOrRd', linewidths=0.5)
    
    plt.title('Incident Heatmap: Hour of Day vs. Day of Week', fontsize=16)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Hour of Day (0-23)', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(output_dir / '2_hourly_heatmap.png', dpi=300)
    plt.close()

    #the heapmap allows us to visualize the distribution of incidents across different hours and days, highlighting peak times.
    #The heat map shows the most of the accidents happen on Tuesday, Wednesday & Thursday at 10PM which is an unexpected insight

def perform_anova_test(df):
    """
    Statistical Test (ANOVA): Analysis of Variance Test
    ANOVA is perfect for comparing counts or averages across groups (like incident volume per hour)
    Are incident volumes significantly different across different hours of the day?
    """
    print("\n--- STATISTICAL TEST 1: ANOVA ---")
    print("Hypothesis: Do incident counts differ significantly by hour of the day?")
    
    # Get daily counts for each specific hour
    hourly_daily_counts = df.groupby([df['Published Date'].dt.date, 'published_hour']).size().unstack(fill_value=0)
    
    # Prepare data for ANOVA (list of arrays, one for each hour)
    hour_data = [hourly_daily_counts[hour].values for hour in range(24) if hour in hourly_daily_counts.columns]
    
    # Run One-Way ANOVA
    f_stat, p_value = stats.f_oneway(*hour_data)
    
    print(f"F-Statistic: {f_stat:.2f}")
    print(f"P-Value: {p_value:.2e}")
    
    if p_value < 0.05:
        print("Conclusion: REJECT null hypothesis. Incident volumes are significantly different depending on the hour of the day.")
    else:
        print("Conclusion: FAIL TO REJECT null hypothesis. No significant difference across hours.")

    """RESULTS of ANOVA TEST: "Visually, I saw a spike in incidents mid-week at 10 PM. 
    I wanted to be mathematically rigorous, so I ran an ANOVA test, got a p-value near zero, 
    and rejected the null hypothesis—proving that the time of day truly does impact incident volume."""

def perform_chi_square_test(df):
    """
    Statistical Test (Chi-Square): Used for comparing categorical distributions (like incident types across years)
    Did the distribution of incident types change significantly between 2023 and 2026?
    """
    print("\n--- STATISTICAL TEST 2: CHI-SQUARE ---")
    print("Hypothesis: Is the type of incident dependent on the year (2023 vs 2026)?")
    
    # Filter for just 2023 and 2026
    df_compare = df[df['published_year'].isin([2023, 2026])]
    
    # Create a contingency table (Year vs. Incident Type)
    contingency_table = pd.crosstab(df_compare['published_year'], df_compare['Issue Reported'])
    
    # Run Chi-Square test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"Chi2-Statistic: {chi2_stat:.2f}")
    print(f"P-Value: {p_value:.2e}")
    
    if p_value < 0.05:
        print("Conclusion: REJECT null hypothesis. The distribution of incident types changed significantly between 2023 and 2026.")
    else:
        print("Conclusion: FAIL TO REJECT null hypothesis. Incident type proportions remained stable.")

if __name__ == "__main__":
    # Define file paths relative to this script being in the src/ folder
    base_dir = Path(__file__).resolve().parent.parent
    input_file = base_dir / 'data' / 'processed' / 'incidents_features.csv'
    output_dir = base_dir / 'output' / 'visualizations'
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists():
         print(f"Error: Could not find {input_file}.")
         print("Make sure you ran clean_transform.py and feature_engineering.py first!")
    else:
        df = load_data(input_file)
        
        # 1. Visualizations
        analyze_daily_trend(df, output_dir)
        analyze_hourly_heatmap(df, output_dir)
        
        # 2. Statistical Tests
        perform_anova_test(df)
        perform_chi_square_test(df)
        
        print(f"\nPhase A Complete! Visualizations saved to {output_dir}")