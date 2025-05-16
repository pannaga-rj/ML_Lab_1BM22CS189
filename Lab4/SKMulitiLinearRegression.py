import pandas as pd
# Step 2 : import data
house = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Boston.csv')

# display first 5 rows
house.head()


y = house['MEDV']

X = house.drop(['MEDV'],axis=1)

# Step 4 : train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)

# Step 5 : select model
from sklearn.linear_model import LinearRegression
model = LinearRegression()


# Step 6 : train or fit model
model.fit(X_train,y_train)


model.intercept_

model.coef_


