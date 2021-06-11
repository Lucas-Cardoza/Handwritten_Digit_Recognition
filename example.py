# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation = 'sigmoid'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer = 'SGD', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs = 5)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)

probability = model.predict(test_images)

#for i in range(5):
 #   print(probability[i])

plt.figure(figsize = (10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid([False])
    plt.imshow(test_images[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[test_labels[i]])
plt.show()


print('\nTest accuracy: ', test_acc)
