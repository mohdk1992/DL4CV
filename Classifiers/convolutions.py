from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2


# Convolution function
def convolve(image, K):
    [im_r, im_c] = image.shape[:2]
    [k_r, k_c] = K.shape[:2]
    pad = (k_r - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((im_r, im_c), dtype="float")
    for i in range(im_r):
        for j in range(im_c):
            roi = image[i: i + (pad * 2) + 1, j: j + (pad * 2) + 1]
            k = np.sum(roi * K)
            output[i, j] = k
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
args = vars(ap.parse_args())

# Kernels
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
bigBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
sharpen = np.array(([0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]), dtype="int")
laplacian = np.array(([0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]), dtype="int")
kernelBank = (("small_blur", smallBlur),
              ("big_blur", bigBlur),
              ("sharpen", sharpen),
              ("laplacian", laplacian))

# Read image
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply kernels
for (kernel, K) in kernelBank:
    print("[INFO] applying {} Kernel".format(kernel))
    output = convolve(gray, K)

    cv2.imshow("Original", gray)
    cv2.imshow("{} - convolve".format(kernel), output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()