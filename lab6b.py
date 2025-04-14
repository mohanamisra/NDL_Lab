import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam

time_steps, input_dim, hidden_dim, output_dim = 5, 3, 4, 2

inputs = np.random.randn(100, time_steps, input_dim) * 0.1
targets = np.random.randn(100, time_steps, output_dim) * 0.1

model = Sequential([
    SimpleRNN(hidden_dim, return_sequences=True, input_shape=(time_steps, input_dim)),
    Dense(output_dim)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

model.fit(inputs, targets, epochs=50, verbose=2)
print(model.evaluate(inputs, targets, verbose=2))
