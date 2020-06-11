from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))


def predict(X, W):
    preds = sigmoid_activation(X.dot(W))
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1
    return preds


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
args = vars(ap.parse_args())

# Generate data
n = 1000
m = 2
(X, y) = make_blobs(n_samples=n, n_features=m, centers=2, cluster_std=1.5, random_state=42)
y = y.reshape((n, 1))
X = np.c_[X, np.ones((n, 1))]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize parameters
W = np.random.randn(m + 1, 1)
losses = []

# Train
for epoch in range(args["epochs"]):
    preds = sigmoid_activation(X_train.dot(W))
    error = preds - y_train
    loss = np.sum(np.square(error))
    losses.append(loss)
    gradient = X_train.T.dot(error)
    W += -args["alpha"] * gradient

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch{}, loss={:.7f}".format(int(epoch + 1), loss))

# Test
print("[INFO] evaluation...")
preds = predict(X_test, W)
print(classification_report(y_test, preds))

# Visualize
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(X_test[:, 0], X_test[:, 1], marker="o", c=y_test, s=30)

plt.style.use("ggplot")
plt.figure()
plt.title("Training loss")
plt.plot(range(0, args["epochs"]), losses)
plt.xlabel("# of Epochs")
plt.ylabel("Loss")
plt.show()
