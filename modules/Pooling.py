import numpy as np
from enums.enums import PoolingMode

class PoolingStage:
    def __init__(
        self,
        filter_size: int,
        stride_size: int = 1,
        mode: int = PoolingMode.POOLING_MAX, # default mode MAX
        inputs: np.ndarray = None
    ) -> None:
        self.inputs: np.ndarray = inputs
        self.input_size = inputs[0].shape[0] if self.inputs else 0
        
        self.filter_size = filter_size
        self.stride_size = stride_size

        self.mode = mode

        self.feature_maps: np.ndarray = []
        self.feature_map_size = (self.input_size - self.filter_size) // self.stride_size + 1

    def __str__(self) -> str:
        return f"\nPOOLING STAGE\n--------\nInput: {self.inputs}\n\nOutput: {self.feature_maps}\n"

    def setInput(self, inputs):
        self.inputs = inputs
        self.input_size = inputs[0].shape[0]
        self.feature_map_size = (self.input_size - self.filter_size) // self.stride_size + 1

    def getOutput(self):
        return self.feature_maps

    def poolingMax(self, input):
        return np.max(input)
    
    def poolingAvg(self, input):
        return np.sum(input)/len(input)

    def convolve(self, input):
        feature_map = np.zeros((self.feature_map_size, self.feature_map_size), dtype=float)

        for i in range(0, self.feature_map_size, self.stride_size) : 
            for j in range(0, self.feature_map_size, self.stride_size) :
                inputSubset = [input[i][j:(j + self.filter_size)] for i in range(i, i + self.filter_size)]
                
                if (self.mode == PoolingMode.POOLING_MAX):
                    feature_map[i][j] = self.poolingMax(inputSubset)

                if (self.mode == PoolingMode.POOLING_AVG):
                    feature_map[i][j] = self.poolingAvg(inputSubset)
        
        return feature_map

    def calculate(self):
        for input in self.inputs:
            feature_map = self.convolve(input)
            self.feature_maps.append(feature_map)

        self.feature_maps = np.array(self.feature_maps)