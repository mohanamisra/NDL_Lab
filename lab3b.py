import numpy as np
import tensorflow as tf

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

for lr in [0.01, 0.1, 0.5]:
    print(f"\nTraining with Learning Rate: {lr}")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='tanh', input_shape=(2,)),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=75, restore_best_weights=True)
    history = model.fit(X, y, epochs=3000, verbose=0, callbacks=[es])

    final_err = history.history['loss'][-1]
    print(f"Stopped early at epoch {len(history.history['loss'])} with error: {final_err}")
    print(f"Final Mean Squared Error: {final_err:.15f}\n\nXOR Gate Results:")
    for i in X:
        pred = model.predict(np.array([i]), verbose=0)[0][0]
        print(f"Input: {list(i)} Predicted Output: [{pred:.2f}]")
