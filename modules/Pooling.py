import numpy as np
from enums.enums import PoolingMode

class PoolingStage:
    def __init__(
        self,
        filter_size: int,
        stride_size: int = 1,
        mode: int = PoolingMode.POOLING_MAX, # default mode MAX
        input: np.ndarray = None
    ) -> None:
        self.input: np.ndarray = input
        self.input_size = input[0].shape[0] if self.input else 0
        
        self.filter_size = filter_size
        self.stride_size = stride_size

        self.mode = mode

        self.feature_maps: np.ndarray = []
        self.feature_map_size = (self.input_size - self.filter_size) // self.stride_size + 1

    def __str__(self) -> str:
        return f"\nPOOLING STAGE\n--------\nInput: {self.input}\n\nOutput: {self.feature_maps}\n"

    def setInput(self, input):
        self.input = input
        self.input_size = input[0].shape[0]
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
        for input in self.input:
            feature_map = self.convolve(input)
            self.feature_maps.append(feature_map)

        self.feature_maps = np.array(self.feature_maps)

    def backprop(self, dL_dOut):
        """
        Backpropagation for the pooling layer.
        
        Parameters:
        - dL_dOut: Gradient with respect to the output of the pooling layer.
        
        Returns:
        - dL_dInput: Gradient with respect to the input of the pooling layer.
        """
        dL_dInput = np.zeros_like(self.input)

        for f in range(self.input.shape[0]):  # Loop over each feature map
            for i in range(0, self.feature_map_size, self.stride_size):
                for j in range(0, self.feature_map_size, self.stride_size):
                    receptive_field = self.input[f, i:i+self.filter_size, j:j+self.filter_size]
                    if self.mode == PoolingMode.POOLING_MAX:
                        max_val_position = np.unravel_index(receptive_field.argmax(), receptive_field.shape)
                        dL_dInput[f, i+max_val_position[0], j+max_val_position[1]] = dL_dOut[f, i//self.stride_size, j//self.stride_size]
                    elif self.mode == PoolingMode.POOLING_AVG:
                        dL_dInput[f, i:i+self.filter_size, j:j+self.filter_size] += dL_dOut[f, i//self.stride_size, j//self.stride_size] / (self.filter_size**2)

        return dL_dInput
    
    def reset(self):
        self.feature_maps = []