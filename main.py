from enums.enums import ActivationFunction, PoolingMode
from layers.Convolution import ConvolutionLayer
from layers.Dense import DenseLayer
from layers.Flatten import FlattenLayer
from Sequential import Sequential

def main() :
    inputMatr = [[[1,2,3], [4,5,6], [7,8,9]]]
    print("Input:")
    print(inputMatr)

    # Create the Convolution Layer
    convLayer = ConvolutionLayer(3, 2, 1, 2)
    denseLayer1 = DenseLayer(16, ActivationFunction.RELU)
    denseLayer2 = DenseLayer(4, ActivationFunction.SIGMOID)

    model = Sequential()
    model.setInput(inputMatr)
    model.addConvLayer(convLayer)
    model.addDenseLayer(denseLayer1)
    model.addDenseLayer(denseLayer2)
    model.forwardProp()

if __name__ == "__main__":
    main()