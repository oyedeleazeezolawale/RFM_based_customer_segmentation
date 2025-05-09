import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def perform_clustering(RFM_table):
    """
    Perform clustering on the RFM table using KMeans clustering algorithm.

    Parameters:
    RFM_table (pd.DataFrame): The RFM table containing the RFM scores.

    Returns:
    pd.DataFrame: The RFM table with cluster labels added.
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(RFM_table[['Recency', 'Frequency', 'Monetary']])

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    RFM_table['Cluster'] = kmeans.fit_predict(scaled_data)

    return RFM_table