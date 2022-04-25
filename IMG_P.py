import cv2
import matplotlib.pyplot as plt
import numpy as np

def process_image(images):
    global kernel

    # print(np.shape(images)[1:])

    # kernel = np.ones((3,3),np.uint8)
    # kernel = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]],np.uint8)
    kernel = np.array([1, 1])
    # kernel = np.array([[0, 0, 1],[0, 0, 1],[0, 0, 1]],np.uint8)
    # kernel = np.array([[1, 1],[1, 1]],np.uint8)

    cut = 1
    # new = np.zeros((len(images),20,20,1))
    # new = np.zeros((len(images),28,28,1))
    new = np.zeros((len(images),28-2*cut,28-2*cut,1))
    # new = np.ones_like(images)

    for i in range(len(images)):
        # print(i)
        new[i] = crop(images[i],cut)
        new[i] = morph(new[i])
        # new[i] = threshold(new[i])
        # new[i] = binarize(new[i])

    return new

def crop(img, c):
    return img[c:-c,c:-c]

    shape = np.shape(img)

    row_first_bool = True
    row_first=0
    row_last=0

    for i in range(shape[0]):
        if any(img[i,:]):
            if row_first_bool:
                row_first = i
                row_first_bool = False
            row_last = i

    col_first_bool = True
    col_first=0
    col_last=0

    # Determining image bounds
    for i in range(shape[0]):
        if any(img[:,i]):
            if col_first_bool:
                col_first = i
                col_first_bool = False
            col_last = i

    count1 = True
    count2 = True
    h = 1+row_last-row_first
    w = 1+col_last-col_first

    #  making the image at least 20x20
    while True:

        if h < 20:
            if count1:
                if row_first > 0:
                    row_first -= 1
                count1 = not count1
            else:
                if row_last < 27:
                    row_last += 1
                count1 = not count1
            h = 1+row_last-row_first

        if w < 20:
            if count2:
                if col_first > 0:
                    col_first -= 1
                count2 = not count2
            else:
                if col_last < 27:
                    col_last += 1
                count2 = not count2

            w = 1+col_last-col_first

        if h == 20 and w == 20:
            # Exit Loop
            break

    return img[row_first:row_last+1,col_first:col_last+1]

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
