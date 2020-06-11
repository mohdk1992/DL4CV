from keras.datasets import cifar10
from pyimagesearch.nn.cnn import MiniVGGNet
from pyimagesearch.callbacks import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="Plot output")
args = vars(ap.parse_args())

# show information on the process ID
print("[INFO process ID: {}".format(os.getpid()))

# Load data
print("[INFO] loading data...")
((X_train, y_train), (X_test, y_test)) = cifar10.load_data()
X_train = X_train.astype("float") / 255.0
X_test = X_test.astype("float") / 255.0
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

# Initialize model
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(32, 32, 3, 10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# construct the set of callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath)]

# Train model
H = model.fit(X_train, y_train, validation_data=(X_test, y_test),
              batch_size=64, epochs=40, callbacks=callbacks, verbose=1)

# Evaluate model
predictions = model.predict(X_test, batch_size=64)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# Final plot
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("# of epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
plt.savefig(args["output"])
