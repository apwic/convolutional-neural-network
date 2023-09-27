from enums.enums import ActivationFunction, PoolingMode, DenseLayerType
from layers.Convolution import ConvolutionLayer
from layers.Dense import DenseLayer
from layers.Flatten import FlattenLayer
from Sequential import Sequential
from utils.ImageToMatrix import ImageToMatrix
import numpy as np

def main() :
    input = np.array([1])

    weights1 = [
        [2.0, 0.0]
    ]

    bias1 = [1, 1]

    layer1 = DenseLayer(2, ActivationFunction.RELU)
    layer1.setInput(input=input)
    layer1.setWeight(weights=weights1)
    layer1.setBiases(biases=bias1)
    layer1.forward()

    output1 = layer1.getOutput()
    print(f"Output 1: {output1}")

    weights2 = [
        [0.5, 0.2, -0.8],
        [0.3, -0.6, 0.4]
    ]

    bias2 = [0.2, 0.3, 0.1]

    target = [2.05, 0.35, 0.05]

    layer2 = DenseLayer(3, ActivationFunction.RELU)
    layer2.setInput(input=output1)
    layer2.setWeight(weights=weights2)
    layer2.setBiases(biases=bias2)
    layer2.forward()

    output2 = layer2.getOutput()
    print(f"Output 2: {output2}")

    backprop2= layer2.backprop_output(target)
    print(f"Backprop 2: {backprop2}")

    backprop1 = layer1.backprop_hidden(backprop2)
    print(f"Backprop 1: {backprop1}")

if __name__ == "__main__":
    main()