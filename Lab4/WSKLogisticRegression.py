import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(X @ weights)
    cost = -(1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    return cost

# Gradient descent
def gradient_descent(X, y, weights, lr, epochs):
    m = len(y)
    for _ in range(epochs):
        h = sigmoid(X @ weights)
        gradient = (1/m) * X.T @ (h - y)
        weights -= lr * gradient
    return weights

# Load binary classification dataset
iris = load_iris()
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2].reshape(-1, 1)

# Add bias term
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize weights
weights = np.zeros((X.shape[1], 1))

# Train
weights = gradient_descent(X_train, y_train, weights, lr=0.1, epochs=1000)

# Predict
y_pred = sigmoid(X_test @ weights)
y_pred_class = (y_pred >= 0.5).astype(int)

# Accuracy
accuracy = (y_pred_class == y_test).mean()
print("Accuracy (from scratch):", accuracy)
