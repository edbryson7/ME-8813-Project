#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import MNIST
from os.path  import join
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def main():

    input_path = './input'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist = MNIST.MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()

    model = init_model()

    train(model, xtrain, ytrain, xtest, ytest)

def init_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


    return model

def train(model, images, labels, test_images, test_labels):
    history = model.fit(images, labels, epochs=10, 
                    validation_data=(test_images, test_labels))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

def showim(image, title):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title(title)
    plt.show()

if __name__=='__main__':
    main()
