""" geospatial_analysis.py
Phase B: Geospatial Analysis for ATX Traffic Incidents
Focus: K-Means clustering to identify traffic hotposts and zone profiles"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from pathlib import Path

#setting plotting style
sns.set_theme(style="whitegrid")

def load_data(filepath):
    print("LOADING DATA FOR GEOSPATIAL ANALYSIS...")
    return pd.read_csv(filepath)

def perform_kmeans_clustering(df, k=6):
    """Apply K-Means clustering to Latitude & Longitude"""
    print(f"\n-- RUNNING K-MEANS CLUSTERING (k={k}) --")

    #1. Extract Latitude and Longitude for clustering
    coords = df[['Latitude', 'Longitude']].values

    #2. initialize & fit KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)

    #3. Assign each incident to a cluster
    df['Cluster'] = kmeans.fit_predict(coords)

    print("Clustering complete! Added 'Cluster' column to the dataframe.")
    return df, kmeans

def plot_clusters(df, kmeans, output_dir):
    """Visualize the clusters on a scatter plot"""
    print("\nGenerating Cluster Map Plot...")

    plt.figure(figsize=(12,10 ))

    #plot the data points, color-coded by cluster
    scatter = plt.scatter(df['Longitude'], df['Latitude'], 
                          c=df['Cluster'], cmap='tab10', alpha=0.3, s=5)
    
    # Plot cluster centers
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 1], centroids[:, 0], 
                marker='X', s=300, c='red', edgecolors='black', linewidths=2, label='Cluster Centers')
    
    plt.title('Austin Traffic Incident Hotspots (K-Means Clustering)', fontsize=16)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_dir / '3_cluster_map.png', dpi=300)
    plt.close()

def analyze_cluster_profiles(df):
    """Print a summary profile for each geographic cluster."""
    print("\n--- ZONE PROFILES ---")
    
    # Group by our new cluster column
    for cluster_id in sorted(df['Cluster'].unique()):
        zone_data = df[df['Cluster'] == cluster_id]
        
        total_incidents = len(zone_data)
        percent_urgent = zone_data['is_urgent'].mean() * 100
        top_incident = zone_data['Issue Reported'].mode()[0]
        
        print(f"Zone {cluster_id}:")
        print(f"  Total Incidents: {total_incidents:,}")
        print(f"  % Urgent:        {percent_urgent:.1f}%")
        print(f"  Top Issue:       {top_incident}")
        print("-" * 25)

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    input_file = base_dir / 'data' / 'processed' / 'incidents_features.csv'
    output_dir = base_dir / 'output' / 'visualizations'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists():
         print(f"Error: Could not find {input_file}.")
    else:
        df = load_data(input_file)
        
        # Run the analysis
        df, kmeans_model = perform_kmeans_clustering(df, k=6)
        plot_clusters(df, kmeans_model, output_dir)
        analyze_cluster_profiles(df)
        
        print(f"\nPhase B Complete! Map saved to {output_dir}")

'''In my orginal project, I used folium to plot a dot on a map for every single accident.
This was a very slow and inefficient way to visualize the data.
In this new version, I used K-Means clustering to group incidents into 6 clusters and
then plotted the cluster centers on a scatter plot. This is much faster and allows us to see the hotspots of incidents in Austin.

STRATEGIC INSIGHT: Ambulances or Tow Trucks can be pre-positioned near the cluster centers (shown as red X's) 
to reduce response times in high-incident areas.'''

