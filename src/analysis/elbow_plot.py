import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# list of k values to analyze
k_values = range(2, 35) # assuming k from 1 to 35
# k_values = range(2, 11) # assuming k from 1 to 10

# store ss (sum of squares) for each k
ss_list = []

for k in k_values:
    
    # read in cluster file
    file_path = os.path.join(os.path.dirname(__file__), f'../../data/outputs/cluster_results_k{k}.csv')
    df = pd.read_csv(file_path)

    # using pca coordinates for distance calculations
    coords = df[['pca_x', 'pca_y']].values

    # compute sum of squares within clusters using pca coordinates
    ss = 0
    
    # for each cluster, compute centroid and sum squared distances
    for cluster_id, cluster_points in df.groupby('cluster')[['pca_x', 'pca_y']]:
        cluster_coords = cluster_points[['pca_x', 'pca_y']].values
        centroid = cluster_coords.mean(axis=0)
        ss += ((cluster_coords - centroid) ** 2).sum()  # sum of squared distances
    ss_list.append(ss)


# --- [ plotting elbow curve ] ---
plt.figure(figsize=(8,5))
plt.plot(k_values, ss_list, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum of Squares (SS)")
plt.title("Choosing k using Elbow Method")
plt.grid(True)
plt.show()