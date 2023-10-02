from enums.enums import ActivationFunction, PoolingMode, DenseLayerType
from layers.Convolution import ConvolutionLayer
from layers.Dense import DenseLayer
from layers.Flatten import FlattenLayer
from Sequential import Sequential
from utils.ImageToMatrix import ImageToMatrix
import numpy as np

def main():
    # Mock input image of shape 4x4
    input_image = np.array([[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]])

    # Convolution Layer
    convLayer = ConvolutionLayer(
        input_size=4,
        filter_size_conv=2,
        number_of_filter_conv=2,
        filter_size_pool=3,
        stride_size_conv=1,
        stride_size_pool=1,
        padding_size=0,
        mode=PoolingMode.POOLING_MAX
    )
    number_of_filter_conv = 2
    filter_size_conv = 2

    # Initialize filters with He Initialization
    mock_filters = np.random.randn(number_of_filter_conv, filter_size_conv, filter_size_conv) * np.sqrt(2. / filter_size_conv**2)
    print(mock_filters)

    convLayer.setWeights(mock_filters)
    convLayer.setInput(input_image)
    convLayer.calculate()
    conv_output = convLayer.getOutput()
    print(f"Convolution Output: \n{conv_output}\n")

    # Flatten Layer
    flattenLayer = FlattenLayer(len(conv_output[0]))
    flattenLayer.setInput(conv_output)
    flattenLayer.flatten()
    flattened_output = flattenLayer.getOutput()
    print(f"Flattened Output: \n{flattened_output}\n")

    # Dense Layer
    denseLayer = DenseLayer(3, ActivationFunction.RELU)
    input_to_dense = 8  # This is based on the flattened output from the pooling layer
    output_from_dense = 3

    # Initialize weights with He Initialization
    mock_weights = np.random.randn(input_to_dense, output_from_dense) * np.sqrt(2. / input_to_dense)

    # Biases can be initialized to a small positive value to avoid dead neurons in ReLU
    mock_biases = np.full(output_from_dense, 0.01)
    denseLayer.setWeight(mock_weights)
    denseLayer.setBiases(mock_biases)
    denseLayer.setInput(flattened_output)
    denseLayer.forward()
    dense_output = denseLayer.getOutput()
    print(f"Dense Output: \n{dense_output}\n")

    # Mock target for backpropagation
    target = np.array([0.5, 0.8, 0.3])

    # Backpropagation through Dense Layer
    dL_dFlattenedOutput = denseLayer.backprop_output(target)
    print(f"Dense Backprop:\n{dL_dFlattenedOutput}\n")

    print(f"Dense Weight Before:\n{denseLayer.weights}\n")
    print(f"Dense Bias Before:\n{denseLayer.biases}\n")
    denseLayer.update_weights_and_biases()
    print(f"Dense Weight After:\n{denseLayer.weights}\n")
    print(f"Dense Bias After:\n{denseLayer.biases}\n")

    # Backpropagation through Flatten Layer
    dL_dConvOutput = flattenLayer.backprop(dL_dFlattenedOutput)
    print(f"Flatten Backprop:\n{dL_dConvOutput}\n")

    # Backpropagation through Convolution Layer
    dL_dInput = convLayer.backprop(dL_dConvOutput)
    print(f"Gradient w.r.t Input: \n{dL_dInput}\n")

    print(f"Conv Weight Before:\n{convLayer.convolutionStage.filters}\n")
    print(f"Conv Bias Before:\n{convLayer.convolutionStage.biases}\n")
    convLayer.update_weights_and_biases()
    print(f"Conv Weight After:\n{convLayer.convolutionStage.filters}\n")
    print(f"Conv Bias After:\n{convLayer.convolutionStage.biases}\n")

if __name__ == "__main__":
    main()
