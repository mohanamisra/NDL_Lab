import tensorflow as tf

# Example input: (seq_len, batch_size, input_size)
x = tf.random.normal((5, 3, 10))  # 5 time steps, 3 batches, 10 input features

# RNN Cell
rnn_cell = tf.keras.layers.SimpleRNNCell(20)
h_t = [tf.zeros((3, 20))]  # initial hidden state

# Single RNN cell forward step
_, h_t[0] = rnn_cell(x[0], h_t)

# RNN forward propagation (manually through sequence)
h_seq = []
for t in range(x.shape[0]):
    _, h_t[0] = rnn_cell(x[t], h_t)
    h_seq.append(h_t[0])
h_seq = tf.stack(h_seq)

# LSTM Cell
lstm_cell = tf.keras.layers.LSTMCell(20)
state = [tf.zeros((3, 20)), tf.zeros((3, 20))]  # (hidden, cell)

# Single LSTM cell forward step
_, state = lstm_cell(x[0], state)
