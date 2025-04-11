import numpy as np
class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def activate(self, inputs):
        ws = np.dot(self.weights, inputs) + self.bias
        return self.sigmoid(ws)

    def train(self, inputs, y, lr):
        y_pred = self.activate(inputs)
        error = y - y_pred
        self.weights += lr * inputs * error
        self.bias += lr * error


n = 3
neuron = Neuron(n)
lr = 0.1
X_train = np.array([
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 0],
    [0, 1, 1]
])
y_train = np.array([0, 1, 1, 0])
X_test = np.array([
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 0],
    [0, 1, 1]
])

iterations = 10000
for iter in range(iterations):
    index = np.random.randint(len(X_train))
    inputs = X_train[index]
    outputs = y_train[index]
    neuron.train(inputs, outputs, lr)

for inputs in X_test:
    output = neuron.activate(inputs)
    print(f"{inputs} GIVES {output}")
