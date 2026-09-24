"""
utility_functions.py

Shared utilities for ATX traffic analysis.
Plotting helpers, constants, and utility functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================

# Set style globally
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Color palette
COLORS = {
    'urgent': '#d62728',        # Red
    'non_urgent': '#2ca02c',    # Green
    'neutral': '#1f77b4',       # Blue
    'accent': '#ff7f0e',        # Orange
    'secondary': '#9467bd',     # Purple
}


# ============================================================================
# INCIDENT CLASSIFICATION
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


def is_urgent(incident_type):
    """Check if an incident type is considered urgent."""
    return incident_type in URGENT_INCIDENTS


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def plot_incident_distribution(df, top_n=10, title='Top Incident Types', figsize=(12, 6)):
    """
    Plot bar chart of top N incident types.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    incident_counts = df['Issue Reported'].value_counts().head(top_n)
    
    # Color urgent incidents differently
    colors = [
        COLORS['urgent'] if inc in URGENT_INCIDENTS else COLORS['neutral']
        for inc in incident_counts.index
    ]
    
    incident_counts.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel('Count')
    ax.set_ylabel('Incident Type')
    ax.set_title(title)
    ax.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(incident_counts):
        ax.text(v + 100, i, str(v), va='center')
    
    plt.tight_layout()
    return fig, ax


def plot_temporal_trend(df, figsize=(14, 6)):
    """
    Plot incident count over time.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Daily incident count
    daily_counts = df.groupby(df['Published Date'].dt.date).size()
    
    ax.plot(daily_counts.index, daily_counts.values, linewidth=1, alpha=0.7)
    ax.fill_between(daily_counts.index, daily_counts.values, alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Incident Count')
    ax.set_title('Daily Traffic Incident Count (2023-2026)')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, ax


def plot_hourly_heatmap(df, figsize=(14, 8)):
    """
    Plot heatmap of incidents by hour of day and day of week.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Pivot: hour × day of week
    heatmap_data = df.pivot_table(
        values='Traffic Report ID',
        index='published_hour',
        columns='published_day_name',
        aggfunc='count',
        fill_value=0
    )
    
    # Reorder columns (Mon-Sun)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data[[col for col in day_order if col in heatmap_data.columns]]
    
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt='g', ax=ax, cbar_kws={'label': 'Incident Count'})
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Hour of Day')
    ax.set_title('Incident Heatmap: Hour × Day of Week')
    
    plt.tight_layout()
    return fig, ax


def plot_urgency_by_hour(df, figsize=(12, 6)):
    """
    Plot percentage of urgent incidents by hour of day.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    hourly_urgency = df.groupby('published_hour')['is_urgent'].agg(['sum', 'count'])
    hourly_urgency['pct_urgent'] = (hourly_urgency['sum'] / hourly_urgency['count'] * 100).round(1)
    
    ax.bar(hourly_urgency.index, hourly_urgency['pct_urgent'], color=COLORS['urgent'], alpha=0.7)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('% Urgent Incidents')
    ax.set_title('Percentage of Urgent Incidents by Hour')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


# ============================================================================
# STATISTICAL HELPERS
# ============================================================================

def summarize_data(df):
    """
    Print basic summary statistics.
    """
    print("=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"\nDataset shape: {df.shape}")
    print(f"Date range: {df['Published Date'].min()} to {df['Published Date'].max()}")
    print(f"Total incidents: {len(df):,}")
    
    print(f"\nIncident type distribution:")
    print(df['Issue Reported'].value_counts().head(10))
    
    print(f"\nUrgent vs Non-Urgent:")
    urgency = df['is_urgent'].value_counts()
    print(f"  Urgent: {urgency.get(1, 0):,} ({urgency.get(1, 0)/len(df)*100:.1f}%)")
    print(f"  Non-urgent: {urgency.get(0, 0):,} ({urgency.get(0, 0)/len(df)*100:.1f}%)")
    
    print(f"\nAgency distribution:")
    print(df['Agency'].value_counts())
    
    print(f"\nStatus distribution:")
    print(df['Status'].value_counts())
    
    print(f"\nResponse time (minutes) - Summary stats:")
    print(f"  Mean: {df['response_time_minutes'].mean():.1f}")
    print(f"  Median: {df['response_time_minutes'].median():.1f}")
    print(f"  Min: {df['response_time_minutes'].min():.1f}")
    print(f"  Max: {df['response_time_minutes'].max():.1f}")
    print(f"  Missing: {df['response_time_minutes'].isna().sum():,}")


if __name__ == '__main__':
    print("Utility functions loaded. Import and use in your analysis scripts.")
