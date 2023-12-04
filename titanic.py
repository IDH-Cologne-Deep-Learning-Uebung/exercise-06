from itertools import _Predicate
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

y = df["Survived"]
x = df.drop("Survived", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.1)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

input_shape = x_train_scaled.shape[1]

model_1 = Sequential([
    Dense(units=10, activation='relu', input_shape=(input_shape,)),
    Dense(units=2, activation='softmax')  
])

model_1.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
model_1.fit(x_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
accuracy_1 = model_1.evaluate(x_test_scaled, y_test)
print(f"\nAccuracy (Model 1): {accuracy_1[1]}")

# Make predictions
y_pred_proba_1 = model_1.predict(x_test_scaled)
y_pred_1 = np.argmax(y_pred_proba_1, axis=1)

# Evaluate precision, recall, and f1 score
precision_1 = precision_score(y_test, y_pred_1)
recall_1 = recall_score(y_test, y_pred_1)
f1_1 = f1_score(y_test, y_pred_1)
print("Precision (Model 1):"+ str(precision_score(y_test, y_pred)))
print("Recall (Model 1):"+ str(recall_score(y_test, _Predicate)))
print("F1 Score (Model 1):"+ str(f1_score(y_test, _Predicate)))
model_2 = Sequential([
    Dense(units=20, activation='sigmoid', input_shape=(input_shape,)),
    Dense(units=10, activation='relu'),
    Dense(units=2, activation='softmax')  
])

model_2.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
model_2.fit(x_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
accuracy_2 = model_2.evaluate(x_test_scaled, y_test)
print(f"\nAccuracy (Model 2): {accuracy_2[1]}")

# Make predictions
y_pred_proba_2 = model_2.predict(x_test_scaled)
y_pred_2 = np.argmax(y_pred_proba_2, axis=1)

# Evaluate precision, recall, and f1 score
precision_2 = precision_score(y_test, y_pred_2)
recall_2 = recall_score(y_test, y_pred_2)
f1_2 = f1_score(y_test, y_pred_2)
print("Precision (Model 2):"+ str(precision_score(y_test, y_pred)))
print("Recall (Model 2):"+ str(recall_score(y_test, y_pred)))
print("F1 Score (Model 2):"+ str(f1_score(y_test, _Predicate)))