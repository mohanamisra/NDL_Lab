import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import Ridge

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
centers = np.array([[0, 0], [1, 1]])
gamma_manual = 2.0  # CUSTOM RBF
gamma_kernel = 1.0  # USED IN KERNEL TRAINING


# ----- MANUAL RBF -----
def rbf(x, c):
    return np.exp(-gamma_manual * np.linalg.norm(x - c) ** 2)


H = np.array([[rbf(x, c) for c in centers] for x in X])

print("\nInput\tFirst Function\tSecond Function")
for x, h in zip(X, H):
    print(f"{x}\t{h[0]:.4f}\t\t{h[1]:.4f}")

plt.figure()
plt.scatter(H[:, 0], H[:, 1], c='orange', s=60)
plt.plot([0, 1], [1, 0], 'b')
plt.scatter([0, 1], [1, 0], marker='x', s=100, c='blue')
plt.xlabel("Hidden Function 1")
plt.ylabel("Hidden Function 2")
plt.title("Hidden Representation of XOR via RBF")
plt.grid()
plt.axis('equal')

# ----- Part B: Training with/without regularization -----
M = rbf_kernel(X, X, gamma=gamma_kernel)
w_noreg = np.linalg.pinv(M) @ y
w_reg = Ridge(alpha=1.0, fit_intercept=False).fit(M, y).coef_
y_pred_noreg = M @ w_noreg
y_pred_reg = M @ w_reg

print("\nPredictions without regularization:", y_pred_noreg)
print("Predictions with regularization:", y_pred_reg)

plt.figure()
plt.plot(y, 'o-', label='True Output')
plt.plot(y_pred_noreg, 's-', label='Without Regularization')
plt.plot(y_pred_reg, 'x--', label='With Regularization')
plt.xlabel("Data Point Index")
plt.ylabel("Output")
plt.title("Effect of Regularization on XOR Output")
plt.grid()
plt.legend()
plt.show()
