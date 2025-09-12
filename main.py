import numpy as np

np.random.seed(0)

input_1 = [1, 2, 3, 2.5]
input_2 = [2.0, 5.0, -1.0, 2.0]
input_3 = [-1.5, 2.7, 3.3, -0.8]

inputs = [input_1, input_2, input_3]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer_1 = Layer_Dense(4, 5)
layer_2 = Layer_Dense(5, 2)

layer_1.forward(inputs)
layer_2.forward(layer_1.output)

print(layer_2.output)
