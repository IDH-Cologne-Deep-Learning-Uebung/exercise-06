import pandas as pd
from sklearn.model_selection import train_test_split


# read the data from a CSV file (included in the repository)
df = pd.read_csv("exercise-06/data/train.csv")

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

#print("x", df)

# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.

#from tensorflow import keras
#from tensorflow.keras import layers
import keras
from keras import layers
#from keras import ops

#1
model = keras.Sequential()
model.add(layers.Input(shape=(9,)))
#2+3
model.add(layers.Dense(10, activation="sigmoid"))
model.add(layers.Dense(20, activation="relu"))
#model.add(layers.Dense(10, activation="softmax"))
model.add(layers.Dense(1, activation="softmax"))  #out?

#test
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.5)

model.compile(loss="mean_squared_error",optimizer="sgd",metrics=["accuracy"])
#out = model.fit(x_train, y_train, epochs=100, batch_size=5)
model.fit(x_train, y_train, epochs=100, batch_size=10)



#evaluate
y_pred = model.predict(x_train)

#test to (x,1)
y_test = y_test.values.reshape(y_test.shape[0],1)

print(y_test.shape) #shape (x,)
#print(y_test)
print(y_pred.shape) #shape (x ,x)
print
#print(y_pred)

#result = model.evaluate(y_test, y_pred, batch_size=10) #??
#print(result)


# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.

#problem: test und prediction nicht gleich Lang. Wegen split.

from sklearn.metrics import precision_score, recall_score, f1_score

#y_pred = y_pred.reshape(x_test[0],y_pred[1])

print("precision: "+ str(precision_score(y_test, y_pred)))
print("recall: "+ str(recall_score(y_test, y_pred)))
print("f1: "+ str(f1_score(y_test, y_pred)))

#
#ValueError: Found input variables with inconsistent numbers of samples: [286, 428]

#its over 0,56. But it seems it doesn't "learn".





#für model.evaluate
#evaluate problem: Data cardinality is ambiguous:
#  x sizes: 72
#  y sizes: 642
#Make sure all arrays contain the same number of samples.