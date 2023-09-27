import numpy as np
from layers.Convolution import ConvolutionLayer
from layers.Flatten import FlattenLayer
from layers.Dense import DenseLayer

class Sequential:
    def __init__(self) -> None:
        self.conv_layers = []
        self.dense_layers = []
        self.total_layers = 0

        self.input: np.ndarray = None
        self.output: np.ndarray = None

    def setInput(self, input: np.ndarray):
        self.input = input

    def addConvLayer(self, layer: ConvolutionLayer):
        self.conv_layers.append(layer)
        self.total_layers += 1

    def addDenseLayer(self, layer: DenseLayer):
        self.dense_layers.append(layer)
        self.total_layers += 1

    def forwardProp(self):
        curr_input = self.input
        for layer in self.conv_layers:
            layer.setInput(curr_input)
            layer.calculate()
            curr_input = layer.getOutput()

        flatten = FlattenLayer(len(curr_input[0]))
        flatten.setInput(curr_input)
        flatten.flatten()
        curr_input = flatten.output

        for layer in self.dense_layers:
            layer.setInput(curr_input)
            layer.forward()
            curr_input = layer.getOutput()
        
        self.output = curr_input

    def train(self):
        pass
