from enums.enums import ActivationFunction
import numpy as np

class DenseLayer:
    def __init__(
        self,
        units: float,
        activation_function: ActivationFunction
    ) -> None:
        self.units = units
        self.activation_function = activation_function
        self.input: np.ndarray = []
        self.output: np.ndarray = []
        self.weights: np.ndarray = []
        self.biases = np.zeros((1, units))
    
    def __str__(self):
        return f"\nDENSE LAYER\n--------\nInput: {self.input}\n\nOutput: {self.output}\n"
    
    def setInput(self, input: np.ndarray):
        self.input = input.ravel()

        if (len(self.weights) == 0):
            self.weights: np.ndarray = np.random.rand(len(input.ravel()), self.units)

    def setWeight(self, weights: np.ndarray):
        self.weights = weights

    def setBiases(self, biases: np.ndarray):
        self.biases = biases

    def getOutput(self):
        return self.output

    def relu(self, val):
        return np.maximum(0, val)
    
    def sigmoid(self, val):
        return 1 / (1 + np.exp(-val))

    def forward(self):
        z = np.dot(self.input, self.weights) + self.biases

        if (self.activation_function == ActivationFunction.RELU):
            self.output = self.relu(z)
        elif (self.activation_function == ActivationFunction.SIGMOID):
            self.output = self.sigmoid(z)
        else:
            raise ValueError("Activation function out of tubes scope yeah")
        print(self)
