import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data.csv")
print(df.head())


# Check missing values
print(df.isnull().sum())

# Drop rows with missing values
df_cleaned = df.dropna()

# Or fill missing values with mean/median
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].median(), inplace=True)

# For nominal categories
df = pd.get_dummies(df, columns=['Gender', 'Country'], drop_first=True)

# For ordinal categories
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
df[['Education_Level']] = encoder.fit_transform(df[['Education_Level']])


from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization (Z-score)
scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

# Min-Max Normalization
minmax = MinMaxScaler()
df[['Age', 'Salary']] = minmax.fit_transform(df[['Age', 'Salary']])



# Using IQR method
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Salary'] >= Q1 - 1.5*IQR) & (df['Salary'] <= Q3 + 1.5*IQR)]


df['Age_Salary_Ratio'] = df['Age'] / df['Salary']



# Drop irrelevant columns
df.drop(['User_ID', 'Name'], axis=1, inplace=True)

# Correlation-based filtering
correlation_matrix = df.corr()
print(correlation_matrix)



from sklearn.model_selection import train_test_split

X = df.drop('Purchased', axis=1)
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
