# NOTE: NOT TESTED

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Fixed AlexNet (MNIST-optimized)
def create_alexnet():
    return models.Sequential([
        # Conv Block 1 (input: 32x32x1)
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        layers.MaxPooling2D((2, 2)),  # output: 16x16x32

        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),  # 16x16x64
        layers.MaxPooling2D((2, 2)),  # output: 8x8x64

        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),  # 8x8x128

        # Flatten output: 8*8*128 = 8192
        layers.Flatten(),

        # Adjusted Dense Layers
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])


# Load and resize MNIST to 32x32 (AlexNet expects minimum 32x32)
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = tf.image.resize(np.expand_dims(x_train, -1), [32, 32]).numpy() / 255.
x_test = tf.image.resize(np.expand_dims(x_test, -1), [32, 32]).numpy() / 255.

# Create and train model
model = create_alexnet()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train with reduced batch size
history = model.fit(x_train, y_train,
                    epochs=1,
                    batch_size=64,  # Reduced from 128
                    validation_split=0.1)

# Evaluation
plt.figure(figsize=(8, 6))
plt.imshow(tf.math.confusion_matrix(y_test,
                                    np.argmax(model.predict(x_test), axis=1)),
           cmap='Blues')
plt.title('AlexNet Lite - MNIST Results')
plt.colorbar()
plt.show()
