from pyimagesearch.nn import Preceptron
import numpy as np
# Show that one layer nn cannot solve nonlinear problems


# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Def nn + train
p = Preceptron(X.shape[1], alpha = 0.1)
p.fit(X, y, epochs=20)

# Test
for (x, target) in zip(X, y):
    pred = p.predict(x)
    print("[INFO] data={}, truth={}, prediction={}".format(x, target, pred))
