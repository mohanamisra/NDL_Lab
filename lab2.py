import numpy as np


class HebbianNeuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs)

    def activate(self, inputs):
        ws = np.dot(self.weights, inputs)
        return ws

    def train(self, inputs, lr):
        y_pred = self.activate(inputs)
        self.weights += lr * y_pred * inputs


n = 3
neuron = HebbianNeuron(n)
lr = 0.01
iterations = 10000
X_train = np.array([
    0.5, 0.3, 0.2
])
for i in range(iterations):
    index = np.random.randint(len(X_train))
    inputs = X_train[index]
    neuron.train(inputs, lr)

print(neuron.weights)
