from modules import Convolutional, Detector

def main() :
    inputMatr = [[1,2,3], [4,5,6], [7,8,9]]
    print("Input:")
    print(inputMatr)

    print("\n\nCONVOLUTIONAL STAGE\n--------")
    conv = Convolutional.ConvolutionalStage(3, 2, 1)
    conv.resetParams()
    conv.setInput(inputMatr)
    conv.calculate()
    print("Filters:")
    print(conv.filters)
    print("\nFeature maps:")
    print(conv.feature_maps)
    
    print("\n\nDETECTOR STAGE\n--------")
    detector = Detector.DetectorStage(conv.feature_maps)
    detector.detect()
    print("Processed feature maps:")
    print(detector.processed_feature_maps)
    
if __name__ == "__main__":
    main()