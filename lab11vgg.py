# NOTE: REQUIRES INTERNET ACCESS TO DOWNLOAD MODEL WEIGHTS

import matplotlib
matplotlib.use('TkAgg')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load & preprocess MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.image.resize(np.repeat(x_train[..., None]/255., 3, -1), [32,32])
x_test = tf.image.resize(np.repeat(x_test[..., None]/255., 3, -1), [32,32])

# VGG16 base (frozen)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
base_model.trainable = False

# Full model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))

# Confusion matrix
y_pred = np.argmax(model.predict(x_test), axis=1)
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: VGG16 + Dense")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Feature maps from first conv layer for first 5 images
feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block1_conv1').output)
features = feature_extractor.predict(x_test[:5])

for i in range(5):
    plt.figure(figsize=(12, 2))
    for j in range(10):  # Show 10 feature maps per image
        plt.subplot(1, 10, j+1)
        plt.imshow(features[i, :, :, j], cmap='viridis')
        plt.axis('off')
    plt.suptitle(f'Feature Maps for Test Image {i}')
    plt.show()
