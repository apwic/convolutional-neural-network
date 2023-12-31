from enums.enums import ActivationFunction, DenseLayerType
import numpy as np

class DenseLayer:
    def __init__(
        self,
        units: float,
        activation_function: ActivationFunction,
        learning_rate: float = 0.01
    ) -> None:
        self.units = units
        self.activation_function = activation_function
        self.input_size: None
        self.input: np.ndarray = []
        self.output: np.ndarray = []
        self.weights: np.ndarray = []
        self.biases = np.zeros((1, units))

        #backprop attributes
        self.delta_w = np.zeros((len(self.input), units))
        self.delta_b = np.zeros((1, units))
        self.net: np.ndarray = []

        self.learning_rate = learning_rate
    
    def __str__(self):
        return f"\nDENSE LAYER\n--------\nInput: {self.input}\n\nOutput: {self.output}\n"
    
    def setInputSize(self, input_size):
        self.input_size = input_size

    def setInput(self, input: np.ndarray):
        self.input = input.ravel()
        self.input_size = len(self.input)
        self.delta_w = np.zeros((len(input.ravel()), self.units))

        if (len(self.weights) == 0):
            self.weights = np.random.randn(len(input.ravel()), self.units) * np.sqrt(2. / len(input.ravel()))

    def setWeight(self, weights: np.ndarray):
        self.weights = weights

    def setBiases(self, biases: np.ndarray):
        self.biases = biases

    def setLearningRate(self, learning_rate: float):
        self.learning_rate = learning_rate

    def getOutput(self):
        return self.output
    
    def getOutputShape(self):
        return (1, self.units)
    
    def getParamsCount(self):
        return self.input_size * self.units
    
    def getData(self):
        return {
            "type": "dense",
            "params": {
                "units": self.units,
                "activation_function": str(self.activation_function),
                "learning_rate": self.learning_rate,
                "kernel": self.weights.tolist(),
                "biases": self.biases.tolist()
            }
        }

    def relu(self, val, deriv=False):
        if (deriv):
            return np.where(val <= 0, 0, 1)
        
        return np.maximum(0, val)
    
    def sigmoid(self, val, deriv=False):
        val = np.clip(val, -709, 709)  # Clip values to avoid overflow
        if (deriv):
            sigmoid_x = 1 / (1 + np.exp(-val))
            return sigmoid_x * (1 - sigmoid_x)
        return 1 / (1 + np.exp(-val))

    def forward(self):
        z = np.dot(self.input, self.weights) + self.biases
        self.net = z

        if (self.activation_function == ActivationFunction.RELU):
            self.output = self.relu(z)
        elif (self.activation_function == ActivationFunction.SIGMOID):
            self.output = self.sigmoid(z)
        else:
            raise ValueError("Activation function out of tubes scope yeah")

    def dE_dO(self, target):
        return (-(target - self.output))
    
    def dE_dO_BCE(self, y_true):
        epsilon = 1e-15  # to prevent division by zero
        y_pred = np.clip(self.output, epsilon, 1 - epsilon)
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)

    def dO_dNet(self, net):
        if (self.activation_function == ActivationFunction.RELU):
            return self.relu(net, True)
        elif (self.activation_function == ActivationFunction.SIGMOID):
            return self.sigmoid(net, True)
        else:
            raise ValueError("Activation function out of tubes scope yeah")

    def backprop_output(self, target):
        dE_dO = self.dE_dO_BCE(target)
        dO_dNet = self.dO_dNet(self.net)
        dNet_dW = self.input

        self.delta_w = dE_dO * dO_dNet * np.transpose(np.array([dNet_dW]))
        self.delta_b = (dE_dO * dO_dNet)
        
        return np.sum((self.delta_b * self.weights), axis=1)
        
    def backprop_hidden(self, dE_dIn):
        dE_dO = dE_dIn
        dO_dNet = self.dO_dNet(self.net)
        dNet_dW = self.input

        self.delta_w = dE_dO * dO_dNet * np.transpose(np.array([dNet_dW]))
        self.delta_b = (dE_dO * dO_dNet)

        return np.sum((self.delta_b * self.weights), axis=1)
    
    def update_weights_and_biases(self):
        self.weights -= self.learning_rate * self.delta_w
        self.biases -= self.learning_rate * self.delta_b

    def resetOutput(self):
        self.output = []

    def resetAll(self):
        self.output = []
        #backprop attributes
        self.delta_w = np.zeros((len(self.input), self.units))
        self.delta_b = np.zeros((1, self.units))
        self.net: np.ndarray = []