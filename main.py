import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Activate_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X, y = spiral_data(100, 3)

layer_1 = Layer_Dense(2, 3)
layer_2 = Layer_Dense(3, 3)

activation_1 = Activate_ReLU()
activation_2 = Activation_Softmax()

layer_1.forward(X)
activation_1.forward(layer_1.output)

layer_2.forward(activation_1.output)
activation_2.forward(layer_2.output)

print(activation_2.output[:5])
