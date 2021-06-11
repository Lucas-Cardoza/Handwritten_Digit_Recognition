#********************************************************************
#
# Student: Lucas Cardoza
# Course: CSC 540 - Introduction to Artificial Intelligence
# Assignment 4: Handwritten Digit Recognition using Neural Networks (Tensorflow)
# Due Date: Wednesday, April 14, 2021, 11:59 PM
# Instructor: Dr. Siming Liu
#
#********************************************************************


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from keras.utils import np_utils


# Image size variables
img_width = 8
img_height = 8

# Number of images to be displayed for final model results 
num_rows = 5
num_cols = 3

# Variables for model
epochs = 5  # Number of iterations the model will run
hiddenLayers = 40 # The number of nodes in the NN

# Define the classifications vector
class_names = ['0', '1', '2', '3']


# Function to graph preditions data
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label]), color=color)


# Function to plot test data 
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Function to pull data from files
def read_data(data_file):
    data = pd.read_csv(data_file, sep = ',')

    attributes = len(data.values[0])

    X = data.values[0:, 0:attributes-1]
    y = data.values[0:, attributes-1]

    return X, y


# This is the main controller function
def main():
    # Load data
    train_images, train_labels = read_data('optdigits-3.tra')
    test_images, test_labels = read_data('optdigits-3.tes')

    # Reshape data into images of 8X8
    train_images = train_images.reshape(train_images.shape[0], img_width, img_height, 1)
    test_images = test_images.reshape(test_images.shape[0], img_width, img_height, 1)
    
    # Set image shape
    input_shape = (img_width, img_height, 1)

    # Convert values to floats
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    # Normalize data
    train_images = train_images / 16.0
    test_images = test_images / 16.0

    # Print to screen first 25 images in training data
    plt.figure(figsize = (8, 8))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid([False])
        plt.imshow(train_images[i], cmap = plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

    # Define the training model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape = input_shape),
        tf.keras.layers.Dense(hiddenLayers, activation = 'sigmoid'),
        tf.keras.layers.Dense(len(class_names), activation = 'softmax')
    ])

    # Complie the model
    model.compile(optimizer = 'SGD', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])

    # Train the model on training data
    model.fit(train_images, train_labels, epochs = epochs)

    # Evaluate the model with test data
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)

    # Find the model's probabilities/preditions of test data
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)

    # Print first 4 model predictions to console
    for i in range(5):
        print("\nPrediction #", i, ":")
        print(predictions[i])
        print("\nThe prediction is:", np.argmax(predictions[i]))
        print("The actual number is:", test_labels[i])

    # Print the accuracy of the model predictions
    print('\nTest accuracy:', test_acc, "\n")

    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2 *num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i + 1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


