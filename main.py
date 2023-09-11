from Convolutional import * 

def main() :
    inputMatr = [[1,2,3], [4,5,6], [7,8,9]]
    conv = ConvolutionalStage(3, 2, 1)

    conv.resetParams()
    print(conv.filters)
    conv.setInput(inputMatr)
    conv.calculate()
    print(conv.feature_maps)
    
if __name__ == "__main__":
    main()