data = {"USN" : ['1', "2", "3"], "Name" : ["A", "B", "C"]}
df = pd.DataFrame(data)
df


from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names
                  )
df.head()




