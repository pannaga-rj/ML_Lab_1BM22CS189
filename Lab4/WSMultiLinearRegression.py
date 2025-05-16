import numpy as np

# Sample dataset (X: features, y: target)
# Let's predict y from X1 and X2
X = np.array([
    [1, 2104, 5],
    [1, 1416, 3],
    [1, 1534, 3],
    [1, 852, 2]
])  # Add 1 for bias term (intercept)

y = np.array([460, 232, 315, 178])

# Compute beta = (X^T X)^-1 X^T y
X_transpose = X.T
beta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

print("Coefficients (including intercept):")
print(beta)

# Predict for a new sample: [1, 1200, 3] (with intercept)
X_new = np.array([1, 1200, 3])
y_pred = X_new.dot(beta)
print("Prediction for input [1200 sqft, 3 rooms]:", y_pred)
