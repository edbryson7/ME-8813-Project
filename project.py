#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
import matplotlib
import MNIST
from os.path import join
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from sklearn import neighbors
from heatmapplot import *

def main():

    mode = input('select mode. mode 1: train cnn. mode 2: test cnn. mode 3: knn.\n')

    seed(1)
    tf.random.set_seed(2)

    input_path = './input'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist = MNIST.MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()

    # Vectorizing images for KNN
    xktrain = [x.flatten() for x in xtrain]
    xktest = [x.flatten() for x in xtest]

    # Normalizing images for CNN
    xtrain = np.expand_dims(xtrain, axis=-1) # <--- add batch axisp
    xtrain = xtrain.astype('float32') / 255
    xtest = np.expand_dims(xtest, axis=-1) # <--- add batch axisp
    xtest = xtest.astype('float32') / 255

    try:
        mode = int(mode)

    except:
        print('Wrong input')
        return

    if mode == 1:
        model = init_cnn()

        model = train_cnn(model, xtrain, ytrain, xtest, ytest)

    elif mode == 2:
        model = tf.keras.models.load_model('./model')
        model.summary()

        test_cnn(model, xtest,ytest)

    elif mode == 3:
        knn = train_knn(xktrain, ytrain, 2)
        test_knn(knn, xktest, ytest)

def init_cnn():
    print('='*90)
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))

    # model.summary()

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model


def train_cnn(model, images, labels, test_images, test_labels):
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
    # plt.show()

    return model

def test_cnn(model, test_images, test_labels):
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

    fig, ax = plt.subplots()
    plt.title('CNN Confusion Matrix of Test Set Validation')
    plt.xlabel('Correct Digit')
    plt.ylabel('Predicted Digit')

    im, cbar = heatmap(heat, range(10), range(10), ax=ax, cmap='cividis',
            cbarlabel='correct predictions')

    texts = annotate_heatmap(im,heat,textcolors=('white','black'))
    fig.tight_layout()
    plt.savefig('cnn_heatmap.png', dpi=1200)
    plt.show()

def train_knn(train_images, train_labels, k=3):
    knn = neighbors.KNeighborsClassifier(k,weights='distance')
    knn.fit(train_images, train_labels)
    return knn
    
def test_knn(knn, test_images, test_labels):
    y = knn.predict(test_images) 

    assert len(y)==len(test_labels)

    # Creating a heatmap
    heat = np.zeros((10,10),dtype=int)
    for i in range(len(y)):
        heat[y[i],test_labels[i]]+=1

    fig, ax = plt.subplots()
    plt.title('KNN Confusion Matrix of Test Set Validation')
    plt.xlabel('Correct Digit')
    plt.ylabel('Predicted Digit')

    im, cbar = heatmap(heat, range(10), range(10), ax=ax, cmap='cividis',
            cbarlabel='correct predictions')

    texts = annotate_heatmap(im,heat,textcolors=('white','black'))
    fig.tight_layout()
    plt.savefig('knn_heatmap.png', dpi=1200)
    plt.show()

def show_im(image, title):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title(title)
    plt.show()

if __name__=='__main__':
    main()
