preds = sigmoid_activation(X_train.dot(W))
error = preds - y_train
loss = np.sum(np.square(error))
losses.append(loss)
gradient = X_train.T.dot(error)
print(gradient.shape)
W += -args["alpha"] * gradient