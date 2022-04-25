import cv2
import matplotlib.pyplot as plt
import numpy as np

def process_image(images):
    global kernel
    kernel = np.ones((3,3),np.uint8)
    # kernel = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]],np.uint8)
    kernel = np.array([[0, 1],[1, 1]],np.uint8)
    new = np.full_like(images,0)
    for i in range(len(images)):
        # plt.imshow(images[i])
        # plt.show()

        images[i] = binarize(images[i])

        # plt.imshow(new[i])
        # plt.show()

        new[i] = erode(images[i])

    return new

def binarize(image):
    for j in range(len(image)):
        for k in range(len(image)):
            image[j][k] = (lambda x: 1 if x>0 else 0)(image[j][k])
    return image

def erode(image):
    eroded = cv2.erode(image, kernel, iterations = 1)
    return np.expand_dims(eroded, axis=-1) # <--- add batch axisp
