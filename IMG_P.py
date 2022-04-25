import cv2
import matplotlib.pyplot as plt
import numpy as np

def process_image(images):
    global kernel
    kernel = np.ones((3,3),np.uint8)
    # kernel = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]],np.uint8)
    kernel = np.array([[0, 1],[1, 1]],np.uint8)
    # new = np.full_like(images,0)
    for i in range(len(images)):
        # plt.imshow(images[i])
        # plt.show()

        images[i] = binarize(images[i])

        # plt.imshow(new[i])
        # plt.show()

        # images[i] = erode(images[i])

    return images

def binarize(image):
    idx = image < .4
    # image[idx] = 0
    # return image

    return (.3<image)

def erode(image):
    eroded = cv2.erode(image, kernel, iterations = 1)
    return np.expand_dims(eroded, axis=-1) # <--- add batch axisp
