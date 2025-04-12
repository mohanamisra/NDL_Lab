#  RUN ON GPU

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = tf.image.resize(np.expand_dims(X_train, -1), [32, 32]).numpy()/255.0
X_test = tf.image.resize(np.expand_dims(X_test, -1), [32, 32]).numpy()/255.0

def build_model():
  return Sequential([
      Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 1)),
      MaxPooling2D((2, 2)),
      Conv2D(64, (3, 3), padding='same', activation='relu'),
      MaxPooling2D((2, 2)),
      Conv2D(128, (3, 3), padding='same', activation='relu'),
      Flatten(),
      Dense(128, activation='relu'),
      Dropout(0.3),
      Dense(10, activation='softmax'),
  ])

model = build_model()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1, batch_size=128, validation_split=0.1)
y_pred = np.argmax(model.predict(X_test), axis=1)

print(model.evaluate(X_test, y_test, verbose=1))

cm = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(cm, cmap='Blues')
plt.title("AlexNet Confusion Matrix")
plt.colorbar()
plt.show()
