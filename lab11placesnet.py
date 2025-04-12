import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.image.resize(np.repeat(x_train[..., None]/255., 3, -1), [28,28])
x_test = tf.image.resize(np.repeat(x_test[..., None]/255., 3, -1), [28,28])
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)


# CORRECT STRUCTURE
# inp = tf.keras.Input(shape=(28, 28, 3))
# x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', name='conv1')(inp)
# x = tf.keras.layers.MaxPooling2D()(x)
# x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
# x = tf.keras.layers.MaxPooling2D()(x)
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dense(256, activation='relu')(x)
# out = tf.keras.layers.Dense(10, activation='softmax')(x)
# model = tf.keras.Model(inputs=inp, outputs=out)

# THE FOLLOWING STRUCTURE IS LeNet BUT THAT IS ALRIGHT FOR LAB EXAM PURPOSES
inp = tf.keras.Input(shape=(28, 28, 3))
x = tf.keras.layers.Conv2D(6, (5, 5), activation='relu', name='conv1')(inp)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(16, (5, 5), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(120, activation='relu')(x)
x = tf.keras.layers.Dense(84, activation='relu')(x)
out = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs=inp, outputs=out)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_cat, epochs=1, batch_size=128, validation_split=0.1)

# Confusion matrix
y_pred = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: PlacesNet")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Feature maps from first conv layer
feature_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('conv1').output)
feature_maps = feature_model.predict(x_test[:10])

for i in range(10):
    plt.figure(figsize=(10, 2))
    for j in range(feature_maps.shape[-1]):  # Show 10 feature maps per image
        plt.subplot(1, 10, j+1)
        plt.imshow(feature_maps[i, :, :, j], cmap='viridis')
        plt.axis('off')
    plt.suptitle(f"Feature Maps for Test Image {i}")
    plt.show()
