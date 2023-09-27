from enums.enums import ActivationFunction, PoolingMode
from layers.Convolution import ConvolutionLayer
from layers.Dense import DenseLayer
from layers.Flatten import FlattenLayer
from Sequential import Sequential
from utils.ImageToMatrix import ImageToMatrix
import numpy as np

def main() :
    inputMatr = ImageToMatrix()
    inputMatr = np.transpose(inputMatr, (2, 1, 0))

    print("Input:")

    # Create the Convolution Layer
    convLayer = ConvolutionLayer(input_size=256, 
                                 filter_size_conv=4,
                                 number_of_filter_conv=2,
                                 filter_size_pool=3
                                 )

    denseLayer1 = DenseLayer(1, ActivationFunction.RELU)

    model = Sequential()
    model.setInput(inputMatr)
    model.addConvLayer(convLayer)
    model.addDenseLayer(denseLayer1)
    model.forwardProp()

if __name__ == "__main__":
    main()