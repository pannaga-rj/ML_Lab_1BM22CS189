import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris # Import load_iris

# Step 1: Load the Iris dataset directly
iris = load_iris()
# Create a DataFrame from the data and target
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Add the target column for potential reference, though not used for clustering
data['target'] = iris.target


# Step 2: Extract only numeric columns (or select required features)
# All features in the Iris dataset are numeric
X = data[iris.feature_names].values # Use the feature names to select columns

# Step 3: Apply KMeans
# Adjust n_clusters based on the expected number of clusters in your data (3 for Iris)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # Added n_init to suppress future warnings
data['Cluster'] = kmeans.fit_predict(X)

# Step 4: Plot clusters (for 2D data)
# Iris data has 4 features. We will plot the first two features for visualization.
if X.shape[1] >= 2:
    plt.scatter(X[:, 0], X[:, 1], c=data['Cluster'], cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='x', s=200)
    plt.title("K-Means Clustering of Iris Dataset")
    plt.xlabel(iris.feature_names[0]) # Label with actual feature name
    plt.ylabel(iris.feature_names[1]) # Label with actual feature name
    plt.show()
else:
    print("Cannot plot clustering results directly for data with less than 2 features.")
