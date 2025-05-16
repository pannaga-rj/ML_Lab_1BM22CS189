import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import sample

# Step 1: Load CSV
data = pd.read_csv("your_data.csv")
X = data.select_dtypes(include=['float64', 'int64']).values

# Step 2: Choose K clusters
K = 3
n_samples, n_features = X.shape

# Step 3: Initialize centroids randomly from data
centroids = X[sample(range(n_samples), K)]

def euclidean(a, b):
    return np.linalg.norm(a - b)

# Step 4–6: Repeat until convergence
for _ in range(100):  # Max 100 iterations
    # Assign clusters
    clusters = [[] for _ in range(K)]
    for idx, point in enumerate(X):
        distances = [euclidean(point, c) for c in centroids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(idx)
    
    # Save old centroids
    old_centroids = centroids.copy()
    
    # Step 5: Recalculate centroids
    for i in range(K):
        if clusters[i]:  # avoid division by zero
            centroids[i] = np.mean(X[clusters[i]], axis=0)
    
    # Check for convergence
    if np.allclose(centroids, old_centroids):
        break

# Step 7: Assign final cluster labels
labels = np.zeros(n_samples)
for i, cluster in enumerate(clusters):
    for idx in cluster:
        labels[idx] = i

# Step 8: Visualize (if 2D)
if n_features == 2:
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='x')
    plt.title("K-Means Clustering (Manual)")
    plt.show()
