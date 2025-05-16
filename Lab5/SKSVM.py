from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# For binary classification (class 0 vs 1)
X = X[y != 2]
y = y[y != 2]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train SVM
clf = SVC(kernel='linear')  # Try 'rbf', 'poly', etc.
clf.fit(X_train, y_train)

# Accuracy
print("Test Accuracy:", clf.score(X_test, y_test))
