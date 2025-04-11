import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from minisom import MiniSom

data = np.random.rand(100, 3)

som_size = 10
som = MiniSom(som_size, som_size, 3, sigma=2.0, learning_rate=0.5)
som.random_weights_init(data)
som.train_random(data, 1000)

plt.figure(figsize=(8, 8))
weights = som.get_weights()
plt.imshow(weights)
plt.colorbar()
plt.title("Self-Organizing Map")
plt.show()

# OPTIONAL BELOW
# mapped = np.array([som.winner(d) for d in data])
#
# plt.figure(figsize=(8, 8))
# plt.pcolor(weights, cmap='bone_r')
# plt.colorbar()
# for i, m in enumerate(mapped):
#     plt.plot(m[0]+0.5, m[1]+0.5, 'o', markerfacecolor='None',
#              markeredgecolor='red', markersize=12, markeredgewidth=2)
# plt.title('Data points mapped to SOM')
# plt.show()
#
# print(f"SOM training complete. Map size: {som_size}x{som_size}")
# print(f"Number of data points: {len(data)}")
# print(f"Input dimension: {data.shape[1]}")
