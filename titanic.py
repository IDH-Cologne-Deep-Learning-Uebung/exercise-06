import pandas as pd
import numpy as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

# Data preprocessing steps
df = df.drop(["Name", "PassengerId"], axis=1)

def make_numeric(df):
    df["Sex"] = pd.factorize(df["Sex"])[0]
    df["Cabin"] = pd.factorize(df["Cabin"])[0]
    df["Ticket"] = pd.factorize(df["Ticket"])[0]
    df["Embarked"] = pd.factorize(df["Embarked"])[0]
    return df

df = make_numeric(df)
df = df.dropna()

# Split the data into input features and target variable
y = df["Survived"]
x = df.drop("Survived", axis=1)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.1)

# Define the model
model = Sequential()
model.add(Dense(10, input_shape=(x_train.shape[1],), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy}")

# Make predictions
y_pred = model.predict_step(x_test)

# Evaluate precision, recall, and f1-score
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
