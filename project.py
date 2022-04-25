#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt

import sys
from os.path import join
import random

import MNIST

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from sklearn import neighbors
from sklearn.metrics import accuracy_score
from heatmapplot import *

import IMG_P

def main():
    # seed(1)
    tf.random.set_seed(2)

    try:
        mode = sys.argv[1]
        mode = int(mode)
    except:
        mode = 0
        pass

    input_path = './input'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist = MNIST.MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()

    # Vectorizing images for KNN

    # Normalizing images for CNN
    xtrain = np.expand_dims(xtrain, axis=-1) # <--- add batch axisp
    xtrain = xtrain.astype('float32') / 255
    xtest = np.expand_dims(xtest, axis=-1) # <--- add batch axisp
    xtest = xtest.astype('float32') / 255


    if mode == 1:
        CNN1(xtrain, ytrain, xtest, ytest)
        CNN2(xtest,ytest)

    elif mode == 2:
        CNN2(xtest,ytest)

    elif mode == 3:
        KNN(xtrain, ytrain, xtest, ytest)

    else:
        model.summary()
        CNN1(xtrain, ytrain, xtest, ytest)
        CNN2(xtest,ytest)

        KNN(xtrain, ytrain, xtest, ytest)

def CNN1(xtrain, ytrain, xtest, ytest):
    train_cnn(xtrain, ytrain, xtest, ytest)

def CNN2(xtest, ytest):
    test_cnn(xtest,ytest)

def KNN(xtrain, ytrain, xtest, ytest):
    # xtrain = IMG_P.process_image(xtrain)
    # xtest = IMG_P.process_image(xtest)
    xktrain = [x.flatten() for x in xtrain]
    xktest = [x.flatten() for x in xtest]

    knn = train_knn(xktrain, ytrain, 1)
    test_knn(knn, xktest, ytest)

    show_random(xtest)

def init_cnn():
    print('='*90)
    model = models.Sequential()

    model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model


def train_cnn(images, labels, test_images, test_labels):
    model = init_cnn()
    model.summary()

    print('-'*90)
    history = model.fit(images, labels, epochs=10, batch_size=250, verbose=1,
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

def test_cnn(test_images, test_labels):
    model = tf.keras.models.load_model('./model')
    model.summary()

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print(f'Validation Loss {test_loss}')
    print(f'Validation Accuracy {test_acc}')

    y = model.predict(test_images)

    # Pulling the categorized label
    y = [np.argmax(i) for i in y]

    assert len(y)==len(test_labels)

    # Creating a heatmap
    heat = np.zeros((10,10))#,dtype=int)
    for i in range(len(y)):
        heat[y[i],test_labels[i]]+=1

    for i in range(10):
        heat[:,i]=heat[:,i]/sum(heat[:,i])

    fig, ax = plt.subplots()
    plt.title('CNN Confusion Matrix of Test Set Validation\n')
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

    acc = accuracy_score(test_labels, y)
    print(f'Accuracy: {acc}')

    assert len(y)==len(test_labels)

    # Creating a heatmap
    heat = np.zeros((10,10))#,dtype=float32)
    for i in range(len(y)):
        heat[y[i],test_labels[i]]+=1

    for i in range(10):
        heat[:,i]=heat[:,i]/sum(heat[:,i])

    fig, ax = plt.subplots()
    plt.title('KNN Confusion Matrix of Test Set Validation\n')
    plt.xlabel('Correct Digit')
    plt.ylabel('Predicted Digit')

    im, cbar = heatmap(heat, range(10), range(10), ax=ax, cmap='cividis',
            cbarlabel='correct predictions')

    texts = annotate_heatmap(im,heat,textcolors=('white','black'))
    fig.tight_layout()
    plt.savefig('knn_heatmap.png', dpi=1200)
    plt.show()

def show_random(images):
    ind = [random.randint(0,len(images)) for i in range(10)]
    for i in ind:
        plt.imshow(images[i])
        plt.show()

def show_im(image, title):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title(title)
    plt.show()

if __name__=='__main__':
    main()
