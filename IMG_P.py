import cv2
import matplotlib.pyplot as plt
import numpy as np

def process_image(images):
    global kernel

    # print(np.shape(images)[1:])

    # kernel = np.ones((3,3),np.uint8)
    kernel = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]],np.uint8)
    # kernel = np.array([[0, 0, 1],[0, 0, 1],[0, 0, 1]],np.uint8)
    # kernel = np.array([[1, 1],[1, 1]],np.uint8)

    cut = 1
    new = np.zeros((len(images),28-2*cut,28-2*cut,1))
    # new = np.ones_like(images)

    for i in range(len(images)):
        # new[i] = images[i]
        new[i] = crop(images[i],cut)
        # new[i] = morph(new[i])
        # new[i] = threshold(new[i])

    return new

def crop(image,c):
    return image[c:-c,c:-c]

def threshold(image):
    idx = image < .2
    image[idx] = 0
    return image

def binarize(image):
    return (.8<image)

def morph(image):
    image = cv2.dilate(image, kernel, iterations = 1)
    image = cv2.erode(image, kernel, iterations = 1)
    return np.expand_dims(image, axis=-1) # <--- add batch axisp
