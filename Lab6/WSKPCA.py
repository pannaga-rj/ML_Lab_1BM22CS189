import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("your_data.csv")
X = data.select_dtypes(include=['float64', 'int64']).values

# Step 1: Standardize
X_meaned = X - np.mean(X, axis=0)

# Step 2: Covariance matrix
cov_mat = np.cov(X_meaned, rowvar=False)

# Step 3: Eigenvalues and eigenvectors
eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

# Step 4: Sort eigenvectors by eigenvalues in descending order
sorted_index = np.argsort(eigen_values)[::-1]
sorted_eigenvalues = eigen_values[sorted_index]
sorted_eigenvectors = eigen_vectors[:, sorted_index]

# Step 5: Choose top k eigenvectors
k = 2
eigenvector_subset = sorted_eigenvectors[:, 0:k]

# Step 6: Transform the data
X_reduced = np.dot(X_meaned, eigenvector_subset)

# Plot
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.5)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection (Manual)")
plt.show()
