import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LambdaCallback

# Parameters
time_steps, input_dim, hidden_dim, output_dim = 5, 3, 4, 2

# Generate data (same as original)
inputs = np.random.randn(100, time_steps, input_dim) * 0.1
targets = np.random.randn(100, time_steps, output_dim) * 0.1

# Build model in one go
model = Sequential([
    SimpleRNN(hidden_dim, return_sequences=True, input_shape=(time_steps, input_dim)),
    Dense(output_dim)
])

# Compile with same learning rate
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Print callback for same output format
print_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch}, Loss: {logs['loss']:.4f}")
    if epoch % 10 == 0 else None
)

# Train
model.fit(inputs, targets, epochs=50, verbose=0, callbacks=[print_callback])
print(model.evaluate(inputs, targets, verbose=0))
