#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
import MNIST
from os.path  import join
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def main():

    mode = input('select mode. mode 1: train model. mode 2: test model. mode 3: both.\n')

    seed(1)
    tf.random.set_seed(2)

    input_path = './input'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist = MNIST.MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)



    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    xtrain = np.expand_dims(xtrain, axis=-1) # <--- add batch axisp
    xtrain = xtrain.astype('float32') / 255
    xtest = np.expand_dims(xtest, axis=-1) # <--- add batch axisp
    xtest = xtest.astype('float32') / 255


    try:
        mode = int(mode)
    except:
        print('Wrong input')
        return

    if mode in [1,3]:
        model = init_model()

        model = train(model, xtrain, ytrain, xtest, ytest)


    if mode in [2,3]:
        model = load_model()
        # model.summary()

        test(model, xtest,ytest)

    if mode not in [1,2,3]:
        print('Wrong input')


def init_model():
    print('='*90)
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    # model.summary()

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model

def load_model():
    return tf.keras.models.load_model('./model')

def train(model, images, labels, test_images, test_labels):
    print('-'*90)
    history = model.fit(images, labels, epochs=2, verbose=1,
                    validation_data=(test_images, test_labels))

    model.save('./model')

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    return model

def test(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print(f'Validation Loss {test_loss}')
    print(f'Validation Accuracy {test_acc}')

    y = model.predict(test_images)

    # Pulling the categorized label
    y = [np.argmax(i) for i in y]

    assert len(y)==len(test_labels)

    # Creating a heatmap
    heat = np.zeros((10,10),dtype=int)
    for i in range(len(y)):
        heat[y[i],test_labels[i]]+=1

    # print(heat)

    fig, ax = plt.subplots()
    im = ax.imshow(heat,cmap='copper')


    # Loop over data dimensions and create text annotations.
    for i in range(10):
        for j in range(10):
            text = ax.text(j, i, heat[i, j],
                           ha="center", va="center", color="w")
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.show()


def showim(image, title):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title(title)
    plt.show()

if __name__=='__main__':
    main()
