import numpy as np
from layers.Convolution import ConvolutionLayer
from layers.Flatten import FlattenLayer
from layers.Dense import DenseLayer

class Sequential:
    def __init__(self) -> None:
        self.layers = []
        self.total_layers = 0

        self.input: np.ndarray = None
        self.output: np.ndarray = None

    def setInput(self, input: np.ndarray):
        self.input = input

    def addLayer(self, layer: ConvolutionLayer):
        self.layers.append(layer)

    def forwardProp(self):
        curr_input = self.input
        for layer in self.layers:
            layer.setInputs(curr_input)
            layer.calculate()
            curr_input = layer.getOutput()

        flatten = FlattenLayer(len(curr_input[0]))
        flatten.setInput(curr_input)
        flatten.flatten()

        print(flatten.ouputs)
