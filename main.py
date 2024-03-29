import numpy as np
import pandas as pd
from train import forward_prop
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('./digit-recognizer/train.csv').to_numpy()
np.random.shuffle(data)
X = data[:, 1:] / 255
Y = data[:, 0]
X_train, X_dev = X[1000:], X[:1000]
Y_train, Y_devc = Y[1000:], Y[:1000]

def load_model_pickle(filename="model_params.pkl"):
    with open(filename, 'rb') as file:
        params = pickle.load(file)
    return params

def make_predictions(X, params):
    _, _, _, A2 = forward_prop(*params, X)
    predictions = np.argmax(A2, axis=0)
    return predictions

def test_prediction(index, params, X, Y):
    current_image = X[:, index].reshape((28, 28))
    prediction = make_predictions(X[:, index:index+1], params)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

params = load_model_pickle()

