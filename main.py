from enums.enums import ActivationFunction, PoolingMode, DenseLayerType
from layers.Convolution import ConvolutionLayer
from layers.Dense import DenseLayer
from layers.Flatten import FlattenLayer
from Sequential import Sequential
from utils.ImageToMatrix import getTrainDataset, getTestDataset
import numpy as np

def main():
    model = Sequential()
    train_X, train_y, val_X, val_y = getTrainDataset(split_ratio=0.9)
    test_X, test_y = getTestDataset()

    conv_layer1 = ConvolutionLayer(
        num_of_input=3,
        input_size=256,
        filter_size_conv=4,
        number_of_filter_conv=3,
        filter_size_pool=4,
        stride_size_conv=4,
        stride_size_pool=4,
        padding_size=0,
        mode=PoolingMode.POOLING_MAX
    )

    conv_layer2 = ConvolutionLayer(
        num_of_input=3,
        input_size=conv_layer1.getOutputShape()[1],
        filter_size_conv=4,
        number_of_filter_conv=6,
        filter_size_pool=4,
        stride_size_conv=1,
        stride_size_pool=1,
        padding_size=0,
        mode=PoolingMode.POOLING_AVG,
    )

    model.addConvLayer(conv_layer1)
    model.addConvLayer(conv_layer2)
    model.addDenseLayer(DenseLayer(
        units=300,
        activation_function=ActivationFunction.SIGMOID
    ))
    model.addDenseLayer(DenseLayer(
        units=150,
        activation_function=ActivationFunction.SIGMOID
    ))
    model.addDenseLayer(DenseLayer(
        units=50,
        activation_function=ActivationFunction.SIGMOID
    ))
    model.addDenseLayer(DenseLayer(
        units=25,
        activation_function=ActivationFunction.SIGMOID
    ))
    model.addDenseLayer(DenseLayer(
        units=5,
        activation_function=ActivationFunction.SIGMOID
    ))
    model.addDenseLayer(DenseLayer(
        units=1,
        activation_function=ActivationFunction.RELU
    ))
    model.printSummary()

    model.setInput(train_X[:5])
    model.setTargets(train_y[:5])
    model.setBatchSize(1)
    model.setNumEpochs(5)
    model.train()
    
if __name__ == '__main__':
    main()