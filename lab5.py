import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def generate_data(n=1000):
    mean, cov = [0, 0], [[3, 2], [2, 2]]
    return np.random.multivariate_normal(mean, cov, n)


class HebbianNeuron:
    def __init__(self, dim, lr=0.01):
        self.w = np.random.randn(dim)
        self.lr = lr

    def train(self, X, epochs=10):
        for _ in range(epochs):
            for x in X:
                y = np.dot(self.w, x)
                self.w += self.lr * y * x

    def get_normalized_weights(self):
        return self.w / np.linalg.norm(self.w)


data = generate_data()
neuron = HebbianNeuron(dim=data.shape[1])
neuron.train(data)

pca = PCA(n_components=1).fit(data)
w_hebb = neuron.get_normalized_weights()
w_pca = pca.components_[0] / np.linalg.norm(pca.components_[0])

print("Hebbian Weights:", w_hebb)
print("PCA Component:", w_pca)

# Visualization
plt.scatter(*data.T, alpha=0.3, label="Input Data")
plt.quiver(0, 0, *w_hebb, color='r', scale=3, label="Hebbian Direction")
plt.quiver(0, 0, *w_pca, color='g', scale=3, label="PCA Direction")
plt.legend()
plt.title("Hebbian Learning vs PCA")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis('equal')
plt.grid()
plt.show()
