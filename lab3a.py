import numpy as np

w1, w2, b = 0.5, 0.5, -1

def activate(x):
    return 1 if x >= 0 else 0

def train(inputs, desired, epochs, lr):
    global w1, w2, b
    for epoch in range(epochs):
        tot_error = 0
        for i in range(len(inputs)):
            A, B = inputs[i]
            output = activate(w1 * A + w2 * B + b)
            error = desired[i] - output
            w1 += lr * error * A
            w2 += lr * error * B
            b += lr * error
            tot_error += abs(error)
        if tot_error == 0:
            break

inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
outputs = np.array([1, 1, 1, 0])
train(inputs, outputs, 100, 0.1)
for i in range(len(inputs)):
    A, B = inputs[i]
    output = activate(w1 * A + w2 * B + b)
    print(f'{A, B} GIVES {output}')
