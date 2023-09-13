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
        self.inputs: np.ndarray = []
        self.outputs: np.ndarray = []
        self.weights: np.ndarray = []
        self.biases = np.ones((1, units))
    
    def __str__(self):
        return f"\nDENSE LAYER\n--------\nInput: {self.inputs}\n\nOutput: {self.outputs}\n"
    
    def setInput(self, inputs: np.ndarray):
        self.inputs = inputs.ravel()
        self.weights: np.ndarray = np.random.rand(len(inputs.ravel()), self.units)

    def relu(self, val):
        return np.maximum(0, val)
    
    def sigmoid(self, val):
        return 1 / (1 + np.exp(-val))

    def forward(self):
        z = np.dot(self.inputs, self.weights) + self.biases

        if (self.activation_function == ActivationFunction.RELU):
            self.outputs = self.relu(z)
        elif (self.activation_function == ActivationFunction.SIGMOID):
            self.outputs = self.sigmoid(z)
        else:
            raise ValueError("Activation function out of tubes scope yeah")
