from modules import Convolutional, Detector, Pooling, Flatten

def main() :
    inputMatr = [[1,2,3], [4,5,6], [7,8,9]]
    print("Input:")
    print(inputMatr)

    print("\n\nCONVOLUTIONAL STAGE\n--------")
    conv = Convolutional.ConvolutionalStage(3, 2, 1, 2)
    conv.resetParams()
    conv.setInput(inputMatr)
    conv.addPadding()

    conv.calculate()
    print("Filters:")
    print(conv.filters)
    print("\nFeature maps:")
    print(conv.feature_maps)
    
    print("\n\nDETECTOR STAGE\n--------")
    detector = Detector.DetectorStage(conv.feature_maps)
    detector.calculate()
    print("Processed feature maps:")
    print(detector.feature_maps)

    print("\n\nPOOLING STAGE\n--------")
    pooling = Pooling.PoolingStage(detector.feature_maps, filter_size=2, stride_size=1)
    pooling.calculate()
    print("Processed feature maps:")
    print(pooling.feature_maps)

    print("\n\nFLATTEN STAGE\n--------")
    flatten = Flatten.FlattenLayer(pooling.feature_map_size)
    flatten.setInput(pooling.feature_maps)
    flatten.flatten()
    print("Flattened feature maps:")
    print(flatten.ouputs)
    
if __name__ == "__main__":
    main()