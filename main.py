from enums.enums import ActivationFunction, PoolingMode
from layers.Convolution import ConvolutionLayer
from layers.Dense import DenseLayer
from layers.Flatten import FlattenLayer
from Sequential import Sequential

def main() :
    inputMatr = [
                    [
                        [4,1,3,5,3],
                        [2,1,1,2,2],
                        [5,5,1,2,3],
                        [2,2,4,3,2],
                        [5,1,3,4,5]
                    ]
                ]
    
    convWeights = [
        [
            [1,2,3],
            [4,7,5],
            [3,-32,25]
        ],
        [
            [12,18,12],
            [18,-74,45],
            [-92,45,-18]
        ]
    ]

    denseWeight1 = [
        [1,2],
        [3,-4]
    ]

    print("Input:")
    print(inputMatr)

    # Create the Convolution Layer
    convLayer = ConvolutionLayer(input_size=5, 
                                 filter_size_conv=3,
                                 number_of_filter_conv=2,
                                 filter_size_pool=3
                                 )
    convLayer.setWeights(weights=convWeights)

    denseLayer1 = DenseLayer(2, ActivationFunction.RELU)
    denseLayer1.setWeight(denseWeight1)

    model = Sequential()
    model.setInput(inputMatr)
    model.addConvLayer(convLayer)
    model.addDenseLayer(denseLayer1)
    model.forwardProp()

if __name__ == "__main__":
    main()