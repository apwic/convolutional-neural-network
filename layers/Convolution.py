import numpy as np
from enums.enums import PoolingMode
from modules.Convolutional import ConvolutionalStage
from modules.Detector import DetectorStage
from modules.Pooling import PoolingStage

class ConvolutionLayer:
    def __init__(
        self,
        inputs: np.ndarray,
    ) -> None:
        self.convolutionStage: ConvolutionalStage = None
        self.detectorStage: DetectorStage = None
        self.poolingStage: PoolingStage = None

        self.inputs = inputs
        self.output: np.ndarray = None

    def __str__(self) -> str:
        return f"\nCONVOLUTION LAYER\n--------\nInput: {self.inputs}\n\nOutput: {self.output}\n"

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
        self.convolutionStage.setParams()
    
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
        self.convolutionStage.setInput(self.inputs)
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