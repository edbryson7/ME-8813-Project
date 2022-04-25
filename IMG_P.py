import cv2
import matplotlib.pyplot as plt
import numpy as np

def process_image(images):
    global kernel

    # print(np.shape(images)[1:])

    # kernel = np.ones((3,3),np.uint8)
    # kernel = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]],np.uint8)
    # kernel = np.array([[0, 0, 1],[0, 0, 1],[0, 0, 1]],np.uint8)
    kernel = np.array([[1, 1],[1, 1]],np.uint8)

    new = np.zeros((len(images),26,26,1))

    for i in range(len(images)):
        new[i] = crop(images[i])
        new[i] = erode(new[i])
        new[i] = binarize(new[i])

    return new

def crop(image):
    return image[1:-1,1:-1]

def binarize(image):
    # idx = image < .7
    # image[idx] = 0
    return image

    # return (.8<image)

def erode(image):
    image = cv2.dilate(image, kernel, iterations = 1)
    image = cv2.erode(image, kernel, iterations = 1)
    return np.expand_dims(image, axis=-1) # <--- add batch axisp
