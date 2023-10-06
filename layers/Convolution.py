import numpy as np
from enums.enums import PoolingMode
from modules.Convolutional import ConvolutionalStage
from modules.Detector import DetectorStage
from modules.Pooling import PoolingStage

class ConvolutionLayer:
    def __init__(
        self,
        num_of_input: int,
        input_size: int,
        filter_size_conv: int,
        number_of_filter_conv: int,
        filter_size_pool: int,
        stride_size_conv: int = 1,
        stride_size_pool: int = 1,
        padding_size: int = 0,
        mode: int = PoolingMode.POOLING_MAX, # default mode MAX
        learning_rate: float = 0.01
    ) -> None:
        self.convolutionStage: ConvolutionalStage = None
        self.detectorStage: DetectorStage = None
        self.poolingStage: PoolingStage = None

        self.setConvolutionStage(num_of_input, input_size, filter_size_conv, number_of_filter_conv, learning_rate, padding_size, stride_size_conv)
        self.setDetectorStage()

        pooling_num_of_input, pooling_input_size = self.convolutionStage.getOutputShape()
        self.setPoolingStage(pooling_num_of_input, pooling_input_size, filter_size_pool, stride_size_pool, mode)

        self.input: np.ndarray = None
        self.output: np.ndarray = None

        self.learning_rate = learning_rate

    def __str__(self) -> str:
        return f"\nCONVOLUTION LAYER\n--------\nInput: {self.input}\n\nOutput: {self.output}\n"

    def setInput(self, input: np.ndarray):
        self.input = input

    def setWeights(self, weights: np.ndarray):
        self.convolutionStage.setParams(weights=weights)

    def setLearningRate(self, learning_rate: float):
        self.learning_rate = learning_rate

    def setConvolutionStage(
        self,
        num_of_input: int,
        input_size: int,
        filter_size: int,
        number_of_filter: int,
        learning_rate: float,
        padding_size: int = 0,
        stride_size: int = 1
    ):
        self.convolutionStage = ConvolutionalStage(
            num_of_input=num_of_input,
            input_size=input_size,
            filter_size=filter_size,
            number_of_filter=number_of_filter,
            padding_size=padding_size,
            stride_size=stride_size,
            learning_rate=learning_rate
        )

        # randomize the params
        self.convolutionStage.resetParams()
    
    def setDetectorStage(
        self,
    ):
        self.detectorStage = DetectorStage()

    def setPoolingStage(
        self,
        num_of_input: int,
        input_size: int,
        filter_size: int,
        stride_size: int = 1,
        mode: int = PoolingMode.POOLING_MAX # default mode MAX
    ):
        self.poolingStage = PoolingStage(
            num_of_input=num_of_input,
            input_size=input_size,
            filter_size=filter_size,
            stride_size=stride_size,
            mode=mode
        )

    def getOutput(self):
        return self.output
    
    def getOutputShape(self):
        return self.poolingStage.getOutputShape()
    
    def getParamsCount(self):
        return self.convolutionStage.getParamsCount()
    
    def getData(self):
        pooling_stage = "max_pooling2d" if self.poolingStage.mode == PoolingMode.POOLING_MAX else "avg_pooling2d"

        return [
            {
                "type": "conv2d",
                "params": {
                    "num_of_input": self.convolutionStage.num_of_input,
                    "input_size": self.convolutionStage.input_size,
                    "filter_size_conv": self.convolutionStage.filter_size,
                    "number_of_filter_conv": self.convolutionStage.number_of_filter,
                    "filter_size_pool": self.poolingStage.filter_size,
                    "stride_size_conv": self.convolutionStage.stride_size,
                    "stride_size_pool": self.poolingStage.stride_size,
                    "padding_size": self.convolutionStage.padding_size,
                    "mode": str(self.poolingStage.mode),
                    "learning_rate": self.learning_rate,
                    "kernel": self.convolutionStage.filters.tolist(),
                    "bias": self.convolutionStage.biases.tolist()
                }
            },
            {
                "type": pooling_stage,
                "params": {}
            }
        ]

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

    def backprop(self, dE_dIn):
        dE_dPooling = self.poolingStage.backprop(dE_dIn)
        dE_dDetector = self.detectorStage.backprop(dE_dPooling)
        dE_dConv = self.convolutionStage.backprop(dE_dDetector)

        return dE_dConv
    
    def update_weights_and_biases(self):
        self.convolutionStage.update_weights_and_biases()

    def reset(self):
        self.convolutionStage.reset()
        self.detectorStage.reset()
        self.poolingStage.reset()
        self.output = None