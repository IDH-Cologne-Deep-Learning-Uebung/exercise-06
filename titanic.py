import pandas as pd

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df = df.drop("Name", axis=1)
df = df.drop("PassengerId", axis=1)

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
def make_numeric(df):
  df["Sex"] = pd.factorize(df["Sex"])[0]
  df["Cabin"] = pd.factorize(df["Cabin"])[0]
  df["Ticket"] = pd.factorize(df["Ticket"])[0]
  df["Embarked"] = pd.factorize(df["Embarked"])[0]
  return df
df = make_numeric(df)

# 3. Remove all rows that contain missing values
df = df.dropna()

# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
y = df["Survived"]
x = df.drop("Survived", axis=1)
#print(x.shape)
#print(y.shape)
# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.1)


from tensorflow import keras
from tensorflow.keras import layers

#first model, one hidden layer, 10 neurons 
#model = keras.Sequential()
#model.add(layers.Input(shape=(9,)))
#model.add(layers.Dense(10, activation="softmax"))
#model.add(layers.Dense(1, activation="softmax"))

#second model, two hidden layers, first 20, second 10 neurons 
model=keras.Sequential()
model.add(layers.Input(shape=(9,)))
model.add(layers.Dense(20, activation="sigmoid"))
model.add(layers.Dense(10, activation="relu"))
model.add(layers.Dense(1, activation="softmax"))

# compile it
model.compile(loss="mean_squared_error",optimizer ="sgd",metrics =["accuracy"]) #Muesste der F-Score selbst berechnet werden?

# train it
model.fit(x_train , y_train , epochs =20 , batch_size =50)
