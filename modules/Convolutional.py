import numpy as np

class ConvolutionalStage: 
    def __init__(
        self,
        input_size: int,
        filter_size: int,
        number_of_filter: int,
        padding_size: int = 0,
        stride_size: int = 1
    ) -> None:
        self.input_size = input_size
        self.inputs: np.ndarray = None

        self.filter_size = filter_size
        self.number_of_filter = number_of_filter
        self.filters: np.ndarray = None

        self.padding_size = padding_size
        self.stride_size = stride_size

        self.feature_map_size = (self.input_size - self.filter_size + 2 * self.padding_size) // self.stride_size + 1
        self.feature_maps: np.ndarray = []

    def __str__(self) -> str:
        return f"\nCONVOLUTION STAGE\n--------\nInput: {self.inputs}\n\nOutput: {self.feature_maps}\n"

    def setInput(self, input: np.ndarray) :
        self.inputs = input

    def getOutput(self):
        return self.feature_maps
    
    def setParams(self, weight=0.5):
        self.filters = np.full((self.number_of_filter, self.filter_size, self.filter_size), weight)

    def resetParams(self):
        self.filters = np.random.randn(self.number_of_filter, self.filter_size, self.filter_size)

    def addPadding(self):
        padded_inputs = np.zeros((self.input_size + 2 * self.padding_size, self.input_size + 2 * self.padding_size), dtype=float)

        for i in range(0, self.input_size):
            for j in range(0, self.input_size):
                padded_inputs[i + self.padding_size][j + self.padding_size] = self.inputs[i][j]
        
        self.inputs = padded_inputs

        print(self.inputs)

    def convolve(self, input, filter) :
        feature_map = np.zeros((self.feature_map_size, self.feature_map_size), dtype=float)

        for i in range(0, self.feature_map_size, self.stride_size) : 
            for j in range(0, self.feature_map_size, self.stride_size) :
                inputSubset = [input[i][j:(j + self.filter_size)] for i in range(i, i + self.filter_size)]
                feature_map[i][j] = np.sum(np.multiply(inputSubset, filter))
        
        return feature_map

    def calculate(self) :
        for i in range(len(self.inputs)):
            feature_map: np.ndarray
            for filter in self.filters : 
                feature_map = self.convolve(self.inputs[i], filter)

            if (i == 0):
                self.feature_maps.append(feature_map)
            else:
                self.feature_maps += feature_map
        
        self.feature_maps = np.array(self.feature_maps)