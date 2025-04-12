import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images[..., tf.newaxis]/255.0
test_images = test_images[..., tf.newaxis]/255.0

model = models.Sequential([
    layers.Conv2D(6, (5,5), activation='tanh', input_shape=(28,28,1)),
    layers.AveragePooling2D((2,2)),
    layers.Conv2D(16, (5,5), activation='tanh'),
    layers.AveragePooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(120, activation='tanh'),
    layers.Dense(84, activation='tanh'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=1,
                    validation_data=(test_images, test_labels), verbose=1)

print(model.evaluate(test_images, test_labels))
predictions = model.predict(test_images)
cm = tf.math.confusion_matrix(test_labels, tf.argmax(predictions, axis=1))

plt.figure(figsize=(8,6))
plt.imshow(cm, cmap='Blues')
plt.title('LeNet-5 Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
