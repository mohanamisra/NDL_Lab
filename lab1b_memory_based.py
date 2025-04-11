import numpy as np
from collections import Counter


class Neuron:
    def __init__(self, k): self.k = k

    def train(self, X, y): self.X, self.y = X, y

    def predict(self, X):
        def pred(x):
            dists = np.linalg.norm(self.X - x, axis=1)
            k_labels = self.y[np.argsort(dists)[:self.k]]
            return Counter(k_labels).most_common(1)[0][0]

        return np.array([pred(x) for x in X])


# Sample data
X_train = np.array([[0, 1], [1, 2], [4, 5], [3, 4]])
y_train = np.array([0, 0, 1, 1])
X_test = np.array([[5, 6], [2, 3]])

# Predict
neuron = Neuron(k=3)
neuron.train(X_train, y_train)
print(neuron.predict(X_test))
