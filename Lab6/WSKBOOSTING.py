import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load and prepare binary classification dataset
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # 1 for setosa, 0 for others
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision stump class
class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X[:, self.feature_index] < self.threshold] = 0
        else:
            predictions[X[:, self.feature_index] > self.threshold] = 0
        return predictions

# AdaBoost training
def adaboost(X, y, n_clf=10):
    n_samples, n_features = X.shape
    weights = np.ones(n_samples) / n_samples
    classifiers = []

    for _ in range(n_clf):
        clf = DecisionStump()
        min_error = float('inf')

        for feature in range(n_features):
            feature_values = np.unique(X[:, feature])
            for threshold in feature_values:
                for polarity in [1, -1]:
                    pred = np.ones(n_samples)
                    if polarity == 1:
                        pred[X[:, feature] < threshold] = 0
                    else:
                        pred[X[:, feature] > threshold] = 0
                    error = np.sum(weights[pred != y])

                    if error < min_error:
                        clf.polarity = polarity
                        clf.threshold = threshold
                        clf.feature_index = feature
                        min_error = error

        EPS = 1e-10
        clf.alpha = 0.5 * np.log((1.0 - min_error) / (min_error + EPS))
        predictions = clf.predict(X)
        weights *= np.exp(-clf.alpha * y * (2 * predictions - 1))
        weights /= np.sum(weights)

        classifiers.append(clf)
    return classifiers

# Prediction function
def predict(X, classifiers):
    clf_preds = [clf.alpha * (2 * clf.predict(X) - 1) for clf in classifiers]
    y_pred = np.sign(np.sum(clf_preds, axis=0))
    return ((y_pred + 1) // 2).astype(int)

# Train and test
classifiers = adaboost(X_train, y_train, n_clf=10)
y_pred = predict(X_test, classifiers)
accuracy = np.mean(y_pred == y_test)
print("AdaBoost Accuracy (manual):", round(accuracy * 100, 2), "%")
