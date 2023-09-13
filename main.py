from enums.enums import ActivationFunction, PoolingMode
from layers.Convolution import ConvolutionLayer
from layers.Dense import DenseLayer
from layers.Flatten import FlattenLayer
from Sequential import Sequential

def main() :
<<<<<<< Updated upstream
    inputMatr =[[[1,2,3], [4,5,6], [7,8,9]], 
                 [[2,4,6], [8,10,12], [14,16,18]]]
=======
    inputMatr = [[[1,2,3], [4,5,6], [7,8,9]]]
>>>>>>> Stashed changes
    print("Input:")
    print(inputMatr)

    # Create the Convolution Layer
    convLayer = ConvolutionLayer(3, 2, 1, 2)
    denseLayer1 = DenseLayer(16, ActivationFunction.RELU)
    denseLayer2 = DenseLayer(4, ActivationFunction.SIGMOID)

<<<<<<< Updated upstream
    convLayer.setConvolutionStage(
        input_size=3,
        filter_size=2,
        number_of_filter=1,
        padding_size=0,
        stride_size=1
    )
=======
    model = Sequential()
    model.setInput(inputMatr)
    model.addConvLayer(convLayer)
    model.addDenseLayer(denseLayer1)
    model.addDenseLayer(denseLayer2)
    model.forwardProp()
>>>>>>> Stashed changes

if __name__ == "__main__":
    main()