import pandas as pd

df = pd.read_csv("/content/tvmarketing.csv")
df

# Visualise the relationship between the features and the response using scatterplots
df.plot(x='TV',y='Sales',kind='scatter')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df['TV'], df['Sales'], test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train.values.reshape(-1, 1), y_train)


y_train


model.coef_



model.intercept_
