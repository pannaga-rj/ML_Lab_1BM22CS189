import numpy as np
import pandas as pd

# Sample dataset
# Let's simulate a simple linear relationship: y = 3x + 2
X = np.array([1, 2, 3, 4, 5])
y = np.array([5, 8, 11, 14, 17])  # y = 3x + 2

# Reshape X to add a column of ones for the intercept term
X_b = np.c_[np.ones((len(X), 1)), X]  # add x0 = 1 to each instance

# Normal Equation: θ = (Xᵀ X)^(-1) Xᵀ y
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Predictions
X_test = np.array([[0], [6]])
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]  # add intercept
y_pred = X_test_b.dot(theta_best)

print("Learned Parameters (Intercept and Coefficient):", theta_best)
print("Predictions for X_test =", X_test.flatten(), "are", y_pred)
