import numpy as np
from enums.enums import PoolingMode
from modules.Convolutional import ConvolutionalStage
from modules.Detector import DetectorStage
from modules.Pooling import PoolingStage

class ConvolutionLayer:
    def __init__(
        self,
        input_size: int,
        filter_size_conv: int,
        number_of_filter_conv: int,
        filter_size_pool: int,
        stride_size_conv: int = 1,
        stride_size_pool: int = 1,
        padding_size: int = 0,
        mode: int = PoolingMode.POOLING_MAX # default mode MAX 
    ) -> None:
        self.convolutionStage: ConvolutionalStage = None
        self.detectorStage: DetectorStage = None
        self.poolingStage: PoolingStage = None

        self.setConvolutionStage(input_size, filter_size_conv, number_of_filter_conv, padding_size, stride_size_conv)
        self.setDetectorStage()
        self.setPoolingStage(filter_size_pool, stride_size_pool, mode)

        self.input: np.ndarray = None
        self.output: np.ndarray = None

    def __str__(self) -> str:
        return f"\nCONVOLUTION LAYER\n--------\nInput: {self.input}\n\nOutput: {self.output}\n"

    def setInput(self, input: np.ndarray):
        self.input = input

    def setWeights(self, weights: np.ndarray):
        self.convolutionStage.setParams(weights=weights)

    def setConvolutionStage(
        self,
        input_size: int,
        filter_size: int,
        number_of_filter: int,
        padding_size: int = 0,
        stride_size: int = 1
    ):
        self.convolutionStage = ConvolutionalStage(
            input_size=input_size,
            filter_size=filter_size,
            number_of_filter=number_of_filter,
            padding_size=padding_size,
            stride_size=stride_size
        )

        # randomize the params
        self.convolutionStage.resetParams()
    
    def setDetectorStage(
        self,
    ):
        self.detectorStage = DetectorStage()

    def setPoolingStage(
        self,
        filter_size: int,
        stride_size: int = 1,
        mode: int = PoolingMode.POOLING_MAX # default mode MAX
    ):
        self.poolingStage = PoolingStage(
            filter_size=filter_size,
            stride_size=stride_size,
            mode=mode
        )

    def getOutput(self):
        return self.output

    def calculate(self):
        # Calculate for each stage and pass the output
        self.convolutionStage.setInput(self.input)
        self.convolutionStage.calculate()
        print(self.convolutionStage)

        self.detectorStage.setInput(self.convolutionStage.getOutput())
        self.detectorStage.calculate()
        print(self.detectorStage)

        self.poolingStage.setInput(self.detectorStage.getOutput())
        self.poolingStage.calculate()
        print(self.poolingStage)

        # Set the output from pooling stage
        self.output = self.poolingStage.getOutput()
        print(self.output)