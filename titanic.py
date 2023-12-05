import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

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

#Model 1 with a single hidden layer of size 10
model = keras.Sequential()
model.add(layers.Input(shape=(10,)))
model.add(layers.Dense(10, activation='softmax')
model.add(layers.Dense(1, activation='softmax')


#Model 2 with 2 hidden layers of sizes 20 and 10

model = keras.Sequential()
model.add(layers.Input(shape=(10,)))
model.add(layers.Dense(20, activation='sigmoid')
model.add(layers.Dense(10, activation='relu')
model.add(layers.Dense(1, activation='softmax')


#compile it
model.compile(loss='mean_square_error', optimizer='sgd', metrics=['accuracy']

#train it
model.fit(x_train, y_train, epochs=100, batch_size=5)

