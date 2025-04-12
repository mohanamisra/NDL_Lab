import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use

use("TkAgg")


# Neuron class with activation functions
class Neuron:
    def __init__(self, weights, bias, activation):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def activate(self, inputs):
        return self.activation(np.dot(inputs, self.weights) + self.bias)


# All activation functions in one dictionary
activations = {
    "Binary Step": lambda x: 1 if x >= 0 else 0,
    "Linear": lambda x: x,
    "Sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "Tanh": np.tanh,
    "ReLU": lambda x: max(0, x),
    "Leaky ReLU": lambda x, alpha=0.01: x if x >= 0 else alpha * x,
    "Softmax": lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
}


# Update weights with gradient descent
def update_weights(weights, bias, inputs, target, output, lr):
    error = target - output
    return weights + lr * error * inputs, bias + lr * error


# Plot activation function
def plot_activation(func, title):
    x = np.linspace(-10, 10, 100)
    if title == "Softmax":
        y = np.array([func(np.array([xi, 0]))[0] for xi in x])
    else:
        y = [func(xi) for xi in x]
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid()
    plt.show()


# Main execution
if __name__ == "__main__":
    weights = np.array([0.5, -0.5])
    bias = 0.1
    inputs = np.array([1.0, 2.0])
    target = 1
    lr = 0.1

    for name, func in activations.items():
        neuron = Neuron(weights, bias, func)
        output = neuron.activate(inputs)
        print(f"{name} Output: {output}")

        plot_activation(func, name)
