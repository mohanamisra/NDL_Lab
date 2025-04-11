import numpy as np

def train_hopfield(patterns):
    num_neurons = len(patterns[0])
    wt_mat = np.zeros((num_neurons, num_neurons))

    for pattern in patterns:
        pattern = np.array(pattern).reshape(-1, 1)
        wt_mat += pattern @ pattern.T

    np.fill_diagonal(wt_mat, 0)
    return wt_mat


def recall_pattern(wt_mat, X, max_iter=10):
    y = np.array(X)
    for _ in range(max_iter):
        for i in range(len(y)):
            net_input = np.dot(wt_mat[i], y)
            y[i] = 1 if net_input >= 0 else -1

    return y


X = [-1, 1, -1, 1, -1, -1, 1, 1]
wt_mat = train_hopfield([X])
noisy = [-1, -1, 1, 1, -1, -1, -1, 1]
recovered = recall_pattern(wt_mat, noisy)

print(recovered)
