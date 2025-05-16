import pandas as pd
import numpy as np
from collections import Counter
from random import randrange

# Load iris dataset manually
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
data = np.column_stack((X, y))
columns = iris.feature_names + ['target']
df = pd.DataFrame(data, columns=columns)

# Split dataset
def train_test_split(data, test_size=0.2):
    data = data.sample(frac=1).reset_index(drop=True)
    test_count = int(test_size * len(data))
    return data.iloc[test_count:], data.iloc[:test_count]

train_data, test_data = train_test_split(df)

# Gini index calculation
def gini_index(groups, classes):
    n_instances = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0: continue
        score = 0.0
        group_labels = group[:, -1]
        for class_val in classes:
            p = np.sum(group_labels == class_val) / size
            score += p * p
        gini += (1 - score) * (size / n_instances)
    return gini

# Split dataset
def test_split(index, value, dataset):
    left = dataset[dataset[:, index] < value]
    right = dataset[dataset[:, index] >= value]
    return left, right

# Choose best split
def get_split(dataset):
    class_values = list(set(dataset[:, -1]))
    b_index, b_value, b_score, b_groups = None, None, float('inf'), None
    for index in range(dataset.shape[1] - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

# Create terminal node
def to_terminal(group):
    outcomes = group[:, -1]
    return Counter(outcomes).most_common(1)[0][0]

# Recursive split
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del node['groups']
    if len(left) == 0 or len(right) == 0:
        node['left'] = node['right'] = to_terminal(np.vstack((left, right)))
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)

# Build tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# Make prediction
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Build a random forest
def subsample(dataset, ratio):
    n_sample = round(len(dataset) * ratio)
    return dataset.sample(n=n_sample, replace=True).values

def random_forest(train, test, max_depth, min_size, sample_size, n_trees):
    trees = []
    for _ in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test.values]
    return predictions

# Bagging predict
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

# Evaluate accuracy
def accuracy_metric(actual, predicted):
    correct = sum(1 for i in range(len(actual)) if actual[i] == predicted[i])
    return correct / float(len(actual)) * 100.0

# Run forest
n_trees = 5
max_depth = 5
min_size = 1
sample_size = 0.8
predictions = random_forest(train_data, test_data, max_depth, min_size, sample_size, n_trees)
actual = test_data['target'].values
acc = accuracy_metric(actual, predictions)
print("Random Forest Accuracy (manual):", round(acc, 2), "%")
