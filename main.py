from enums.enums import ActivationFunction, PoolingMode
from layers.Convolution import ConvolutionLayer
from layers.Dense import DenseLayer
from layers.Flatten import FlattenLayer
from Sequential import Sequential

def main() :
    inputMatr = [[[1,2,3], [4,5,6], [7,8,9], [7,8,9]], 
                 [[2,4,6], [8,10,12], [14,16,18], [7,8,9]],
                 [[1,2,3], [4,5,6], [7,8,9], [7,8,9]], 
                 [[2,4,6], [8,10,12], [14,16,18], [7,8,9]]]
    print("Input:")
    print(inputMatr)

    # Create the Convolution Layer
    convLayer = ConvolutionLayer(inputs=inputMatr)

    convLayer.setConvolutionStage(
        input_size=4,
        filter_size=2,
        number_of_filter=1,
        padding_size=0,
        stride_size=1
    )

    convLayer.setDetectorStage()

    convLayer.setPoolingStage(
        filter_size=2,
        stride_size=1,
        mode = PoolingMode.POOLING_MAX
    )

    convLayer.calculate()

    print(convLayer)

    flattenLayer = FlattenLayer(input_size=len(convLayer.output))
    flattenLayer.setInput(convLayer.output)
    flattenLayer.flatten()

    print(flattenLayer)

    denseLayer1 = DenseLayer(units=16, activation_function=ActivationFunction.SIGMOID)
    denseLayer1.setInput(flattenLayer.outputs)
    denseLayer1.forward()

    print(denseLayer1)

    denseLayer2 = DenseLayer(units=4, activation_function=ActivationFunction.RELU)
    denseLayer2.setInput(denseLayer1.outputs)
    denseLayer2.forward()

    print(denseLayer2)
    
if __name__ == "__main__":
    main()