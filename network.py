import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.e**(-x))

def relu(x):
    return 0 if x < 0 else x

def cost(y, x):
    return (y-x) ** 2

vec_cost = lambda y, x: sum(np.vectorize(cost)(y, x))

def softmax(inputs):
    denominator = 0
    for input in inputs:
        denominator += math.e ** input
    func = np.vectorize(lambda x: (math.e ** x) / denominator)
    return func(inputs)

class Layer:
    def __init__(self, prev_n: int, n: int, activation_func) -> None:
        self.biases = np.zeros((n,))
        # self.biases = np.arange(2,5,1)
        self.weights = np.zeros((n, prev_n))
        # self.weights = np.arange(1,16,1).reshape((n, prev_n))
        self.vec_activation_func = np.vectorize(activation_func)
        
    def compute_activations(self, prev_activations):
        unbiased = np.matmul(self.weights, prev_activations)
        activation_inputs = np.subtract(unbiased, self.biases)
        return self.vec_activation_func(activation_inputs)


class Network:
    def __init__(self, layer_sizes: list[int], input_size: int, activation_func) -> None:
        layers: list[Layer] = []
        prev_n = input_size
        for n in layer_sizes:
            layers.append(Layer(prev_n, n, activation_func))
            prev_n = n
        self.layers = layers

    def compute_result(self, inputs):
        prev_result = inputs
        for layer in self.layers:
            prev_result = layer.compute_activations(inputs)
        return softmax(prev_result)

expected = np.array([0,3,0,5])
actual = np.array([2,3,1,3])
print(vec_cost(actual, expected))

# compute the cost function on all of data and average
def cost_on_data(network: Network, data):
    total_cost = 0
    for point in data:
        total_cost += network.compute_result(point)
    avg_cost = total_cost / len(data)
    return avg_cost

# TODO: decoding data
# TODO: evolution