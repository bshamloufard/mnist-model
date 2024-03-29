import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def init_params(input_dim, hidden_dim, output_dim):
    W1 = np.random.randn(hidden_dim, input_dim) * 0.01
    b1 = np.zeros((hidden_dim, 1))
    W2 = np.random.randn(output_dim, hidden_dim) * 0.01
    b2 = np.zeros((output_dim, 1))
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0
def softmax(Z):
    e_Z = np.exp(Z - np.max(Z, axis=0))
    return e_Z / e_Z.sum(axis=0)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.shape[0]
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * ReLU_deriv(Z1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return dW1, db1, dW2, db2

def update_params(params, grads, learning_rate):
    W1, b1, W2, b2 = params
    dW1, db1, dW2, db2 = grads
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

def compute_accuracy(X, Y, params):
    W1, b1, W2, b2 = params
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = np.argmax(A2, axis=0)
    accuracy = np.mean(predictions == Y)
    return accuracy

def gradient_descent(X, Y, learning_rate, iterations, hidden_dim):
    input_dim = X.shape[0]
    output_dim = np.max(Y) + 1
    params = init_params(input_dim, hidden_dim, output_dim)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(*params, X)
        grads = backward_prop(Z1, A1, Z2, A2, params[2], X, Y)
        params = update_params(params, grads, learning_rate)
        if i % 10 == 0:
            print(f'Iteration {i}: Accuracy is {compute_accuracy(X, Y, params)}')
                  
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, int(Y.max()) + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T