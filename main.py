import numpy as np

inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

weight_array = [weights1, weights2, weights3]

bias1 = 2
bias2 = 3
bias3 = 0.5

bias_array = [bias1, bias2, bias3]

output = [0, 0, 0]
i, j = 0, 0

while j < len(weight_array):
    while i < len(inputs):
        output[j] += inputs[i] * weight_array[j][i]
        i += 1
    output[j] += bias_array[j]
    i = 0
    j += 1

print(output)
