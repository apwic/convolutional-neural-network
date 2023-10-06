from enums.enums import ActivationFunction, PoolingMode, DenseLayerType
from layers.Convolution import ConvolutionLayer
from layers.Dense import DenseLayer
from layers.Flatten import FlattenLayer
from Sequential import Sequential
from utils.ImageToMatrix import ImageToMatrix
import numpy as np

def main():
    # Mock input image of shape 4x4
    input_image = np.array(
    [
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ],
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]
    ])

    print(input_image.shape)

    target = np.array([0.5, 0.8, 0.3])

    model = Sequential()

    # Convolution Layer
    convLayer = ConvolutionLayer(
        num_of_input=2,
        input_size=4,
        filter_size_conv=2,
        number_of_filter_conv=2,
        filter_size_pool=2,
        stride_size_conv=1,
        stride_size_pool=1,
        padding_size=0,
        mode=PoolingMode.POOLING_MAX
    )
    mock_filters = np.array([
        [[1, 1], [1, 1]],  # A filter that captures bright regions
        [[-1, -1], [-1, -1]]   # A filter that captures dark regions                                                                                        
    ])
    convLayer.setWeights(mock_filters)

    # Dense Layer
    denseLayer = DenseLayer(
        units=3, 
        activation_function=ActivationFunction.RELU
    )
    mock_weights = np.array([
        [0.2, 0.5, -0.3],
        [-0.4, 0.1, 0.6],
        [0.3, -0.2, 0.4],
        [-0.1, 0.3, -0.5],
        [0.2, -0.4, 0.1],
        [-0.3, 0.2, 0.5],
        [0.4, -0.1, -0.2],
        [-0.2, 0.3, 0.4]
    ])
    mock_biases = np.array([0.1, -0.2, 0.3])
    denseLayer.setWeight(mock_weights)
    denseLayer.setBiases(mock_biases)

    batch_size = 1
    num_epochs = 2
    learning_rate = 0.1

    model.addConvLayer(convLayer)
    model.addDenseLayer(denseLayer)
    model.printSummary()
    model.setInput(input_image)
    model.setTargets(target)
    model.setBatchSize(batch_size)
    model.setNumEpochs(num_epochs)
    model.setLearningRate(learning_rate)
    model.train()

    print(f"Dense Weight After:\n{model.dense_layers[0].weights}\n")
    print(f"Dense Bias After:\n{model.dense_layers[0].biases}\n")
    print(f"Conv Weight After:\n{model.conv_layers[0].convolutionStage.filters}\n")
    print(f"Conv Bias After:\n{model.conv_layers[0].convolutionStage.biases}\n")
    
    # model.saveModel('test')
    # model.loadModel('test')
    
if __name__ == "__main__":
    main()
