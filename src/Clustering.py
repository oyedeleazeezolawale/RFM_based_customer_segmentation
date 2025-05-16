import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def perform_clustering(RFM_table):
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(RFM_table[['Recency', 'Frequency', 'Monetary']])

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    RFM_table['Cluster'] = kmeans.fit_predict(scaled_data)

    return RFM_table