from enums.enums import ActivationFunction, PoolingMode, DenseLayerType
from layers.Convolution import ConvolutionLayer
from layers.Dense import DenseLayer
from layers.Flatten import FlattenLayer
from Sequential import Sequential
from utils.ImageToMatrix import getTrainDataset, getTestDataset
import numpy as np

train_X, train_y = getTrainDataset()
test_X, test_y = getTestDataset()

def createModel(load_file = None):
    if (load_file != None):
        model = Sequential()
        model.loadModel(load_file)

        return model

    model = Sequential()

    conv_layer1 = ConvolutionLayer(
        num_of_input=3,
        input_size=train_X[0][0].shape[0],
        filter_size_conv=3,
        number_of_filter_conv=4,  # Reduced number of filters
        filter_size_pool=2,
        stride_size_conv=2,  # Increased stride for dimensionality reduction
        stride_size_pool=1,
        padding_size=1,
        mode=PoolingMode.POOLING_MAX
    )

    conv_layer2 = ConvolutionLayer(
        num_of_input=3,
        input_size=conv_layer1.getOutputShape()[1],
        filter_size_conv=3,
        number_of_filter_conv=2,  # Reduced number of filters
        filter_size_pool=2,
        stride_size_conv=2,  # Increased stride for dimensionality reduction
        stride_size_pool=1,
        padding_size=1,
        mode=PoolingMode.POOLING_MAX,
    )

    model.addConvLayer(conv_layer1)
    model.addConvLayer(conv_layer2)
    model.addDenseLayer(DenseLayer(
        units=16,  # Reduced number of units
        activation_function=ActivationFunction.RELU
    ))
    model.addDenseLayer(DenseLayer(
        units=4,  # Reduced number of units
        activation_function=ActivationFunction.RELU
    ))
    model.addDenseLayer(DenseLayer(
        units=1,
        activation_function=ActivationFunction.SIGMOID
    ))
    
    return model

def run(model, *, num, epoch, batch_size, learning_rate):
    model.setLearningRate(learning_rate)
    model.setInput(train_X[:num])
    model.setTargets(train_y[:num])
    model.setTest(test_X, test_y)
    model.setBatchSize(batch_size)
    model.setNumEpochs(epoch)

    model.train()
    model.test()