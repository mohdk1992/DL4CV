import numpy as np
import cv2

np.random.seed(42)

labels = ["dog", "cat", "panda"]

image = cv2.imread("../datasets/animals/dogs/dogs_00001.jpg")
image_proc = cv2.resize(image, (32,32)).flatten()

W = np.random.randn(3, 3072)
b = np.random.randn(3)
scores = W.dot(image_proc) + b

for (label, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))

cv2.putText(image, "Label: {}".format(labels[np.argmax(scores)]),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
