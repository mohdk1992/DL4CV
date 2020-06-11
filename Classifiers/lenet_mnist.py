from pyimagesearch.nn.cnn import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

# Load data + normalization
print("[INFO] Loading MNIST dataset...")
dataset = fetch_openml('mnist_784')
data = dataset.data.astype("float") / 255.0
data = data.reshape(data.shape[0], 28, 28, 1)
(X_train, X_test, y_train, y_test) = train_test_split(data, dataset.target, test_size=0.25, random_state=42)

# One-hot encoding
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# Initialize model
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
print("[INFO] training model...")
H = model.fit(X_train, y_train, validation_data=[X_test, y_test],
              batch_size=128, epochs=20, verbose=1)

# Evaluate
pred = model.predict(X_test, batch_size=128)
print(classification_report(y_test.argmax(axis=1), pred.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))

# Plot
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("# of epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
