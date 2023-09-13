from enums.enums import PoolingMode
from layers.Convolution import ConvolutionLayer

def main() :
    inputMatr = [[[1,2,3], [4,5,6], [7,8,9]], 
                 [[2,4,6], [8,10,12], [14,16,18]]]
    print("Input:")
    print(inputMatr)

    # Create the Convolution Layer
    convLayer = ConvolutionLayer(inputs=inputMatr)

    convLayer.setConvolutionStage(
        input_size=3,
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
    
if __name__ == "__main__":
    main()