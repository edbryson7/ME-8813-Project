#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
import matplotlib
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

    fig, ax = plt.subplots()

    im, cbar = heatmap(heat, range(10), range(10), ax=ax, cmap='cividis',
            cbarlabel='correct predictions')

    texts = annotate_heatmap(im,heat,textcolors=('white','black'))
    fig.tight_layout()
    plt.show()


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    # if not isinstance(data, (list, np.ndarray)):
        # data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    # threshold = -1
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, data[i, j], **kw)
            texts.append(text)

    return texts

def showim(image, title):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title(title)
    plt.show()

if __name__=='__main__':
    main()
