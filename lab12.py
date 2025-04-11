import numpy as np

def softmax(x): return np.exp(x) / np.sum(np.exp(x), axis=0)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)

def rnn_cell(xt, a_prev, p):
    a = tanh(p["Wax"] @ xt + p["Waa"] @ a_prev + p["ba"])
    y = softmax(p["Wya"] @ a + p["by"])
    return a, y

def rnn_forward(x, a0, p):
    a, y, a_next = [], [], a0
    for t in range(x.shape[2]):
        a_next, yt = rnn_cell(x[:,:,t], a_next, p)
        a.append(a_next); y.append(yt)
    return np.stack(a, 2), np.stack(y, 2)

def lstm_cell(xt, a_prev, c_prev, p):
    z = np.vstack((a_prev, xt))
    ft, it, ot = map(lambda k: sigmoid(p[k] @ z + p['b'+k[1]]), ['Wf','Wi','Wo'])
    cct = tanh(p['Wc'] @ z + p['bc'])
    c = ft * c_prev + it * cct
    a = ot * tanh(c)
    return a, c

# Minimal test
n_x, n_a, n_y, m, T = 3, 5, 2, 4, 6
x = np.random.randn(n_x, m, T)
a0, c0 = np.zeros((n_a, m)), np.zeros((n_a, m))

rnn_p = {k: np.random.randn(*s) for k, s in {
    "Wax": (n_a,n_x), "Waa": (n_a,n_a), "Wya": (n_y,n_a),
    "ba": (n_a,1), "by": (n_y,1)}.items()}

lstm_p = {k: np.random.randn(n_a, n_a+n_x) for k in ['Wf','Wi','Wo','Wc']}
lstm_p.update({f"b{k[1]}": np.random.randn(n_a,1) for k in lstm_p})

a_rnn, y_rnn = rnn_forward(x, a0, rnn_p)
a_lstm, _ = lstm_cell(x[:,:,0], a0, c0, lstm_p)

print("RNN y[0,:,0]:", y_rnn[0,:,0])
print("LSTM a[:,0]:", a_lstm[:,0])
