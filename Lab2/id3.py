import pandas as pd
import numpy as np

# Calculate Entropy
def entropy(data):
    # Calculate the probability of each class
    class_probabilities = data.iloc[:, -1].value_counts(normalize=True)
    return -np.sum(class_probabilities * np.log2(class_probabilities))

# Calculate Information Gain
def information_gain(data, feature):
    total_entropy = entropy(data)
    # Calculate the weighted entropy of the feature
    feature_values = data[feature].unique()
    weighted_entropy = 0
    for value in feature_values:
        subset = data[data[feature] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)
    # Information Gain is the reduction in entropy
    return total_entropy - weighted_entropy

# Find the best feature to split the data
def best_feature(data):
    features = data.columns[:-1]  # Exclude the target column
    gains = {feature: information_gain(data, feature) for feature in features}
    return max(gains, key=gains.get)

# Create the decision tree
def id3(data, features=None):
    # Base case: if all data points belong to the same class
    if len(data.iloc[:, -1].unique()) == 1:
        return data.iloc[:, -1].iloc[0]

    # Base case: if there are no more features to split on
    if len(features) == 0:
        return data.iloc[:, -1].mode()[0]

    # Find the best feature to split on
    best = best_feature(data)
    tree = {best: {}}

    # Remove the best feature from the list
    new_features = features.copy()
    new_features.remove(best)

    # Split the data by the best feature
    for value in data[best].unique():
        subset = data[data[best] == value]
        tree[best][value] = id3(subset, new_features)

    return tree

# Function to classify new examples based on the decision tree
def classify(tree, example):
    if not isinstance(tree, dict):
        return tree
    feature = list(tree.keys())[0]
    value = example[feature]
    return classify(tree[feature][value], example)

# Example usage
# Load dataset into a pandas DataFrame
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'High', 'Low', 'Low', 'High', 'Low', 'Low', 'Low', 'High', 'Low', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Weak', 'Strong', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong', 'Weak'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

# Train the decision tree
tree = id3(data, features=list(data.columns[:-1]))
print("Decision Tree:", tree)

# Classify a new example
example = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'Low', 'Wind': 'Strong'}
prediction = classify(tree, example)
print("Prediction for the example:", prediction)
