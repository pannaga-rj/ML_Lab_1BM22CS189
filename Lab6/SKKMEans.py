import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Load CSV data
data = pd.read_csv("your_data.csv")

# Step 2: Extract only numeric columns (or select required features)
X = data.select_dtypes(include=['float64', 'int64']).values

# Step 3: Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Step 4: Plot clusters (for 2D data)
plt.scatter(X[:, 0], X[:, 1], c=data['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='x', s=200)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
