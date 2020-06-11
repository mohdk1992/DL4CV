from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

import matplotlib.pyplot as plt
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help=("path to the output plot"))
args = vars(ap.parse_args())

# Load data + normalization
print("[INFO] Loading MNIST dataset...")
dataset = fetch_openml('mnist_784')
data = dataset.data.astype("float") / 255.0
(X_train, X_test, y_train, y_test) = train_test_split(data, dataset.target, test_size=0.25)

# One-hot encoding
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# Create nn 784-256-128-10
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

# Train model (SGD)
sgd = SGD(0.01)
model.compile(sgd, loss="categorical_crossentropy", metrics=["accuracy"])
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=128)

# Evaluate
pred = model.predict(X_test, batch_size=128)
print(classification_report(y_test.argmax(axis=1), pred.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

# Plot
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("# of epochs")
plt.ylabel("Loss/Accuracy")
plt.savefig(args["output"])





