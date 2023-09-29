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
        self.input: np.ndarray = None

        self.filter_size = filter_size
        self.number_of_filter = number_of_filter
        self.filters: np.ndarray = None

        self.biases = np.zeros(self.number_of_filter)

        self.padding_size = padding_size
        self.stride_size = stride_size

        self.feature_map_size = (self.input_size - self.filter_size + 2 * self.padding_size) // self.stride_size + 1
        self.feature_maps: np.ndarray = []

        # backprop attribute
        self.delta_filters = None
        self.delta_biases = None

    def __str__(self) -> str:
        return f"\nCONVOLUTION STAGE\n--------\nInput: {self.input}\n\nOutput: {self.feature_maps}\n"

    def setInput(self, input: np.ndarray) :
        self.input = input

    def getOutput(self):
        return self.feature_maps
    
    def setParams(self, weights = np.ndarray):
        self.filters = weights

    def setBiases(self, biases = np.ndarray):
        self.biases = biases

    def resetParams(self):
        self.filters = np.random.randn(self.number_of_filter, self.filter_size, self.filter_size)

    def addPadding(self):
        padded_inputs = np.zeros((self.input_size + 2 * self.padding_size, self.input_size + 2 * self.padding_size), dtype=float)

        for i in range(0, self.input_size):
            for j in range(0, self.input_size):
                padded_inputs[i + self.padding_size][j + self.padding_size] = self.input[i][j]
        
        self.input = padded_inputs

        print(self.input)

    def convolve(self, input, filter, bias) :
        feature_map = np.zeros((self.feature_map_size, self.feature_map_size), dtype=float)

        for i in range(0, self.feature_map_size, self.stride_size) : 
            for j in range(0, self.feature_map_size, self.stride_size) :
                inputSubset = [input[i][j:(j + self.filter_size)] for i in range(i, i + self.filter_size)]
                feature_map[i][j] = np.sum(np.multiply(inputSubset, filter)) + bias
        
        return feature_map

    def calculate(self) :
        for i in range(len(self.input)):
            feature_map = []
            for idx_f in range(len(self.filters)) : 
                feature_map.append(self.convolve(self.input[i], self.filters[idx_f], self.biases[idx_f]))

            if (i == 0):
                self.feature_maps = np.array(feature_map)
            if (i != 0):
                self.feature_maps += feature_map

    def backprop(self, dL_dOut):
        """
        Backpropagation for the convolutional layer.
        
        Parameters:
        - dL_dOut: Gradient with respect to the output of the convolutional layer.
        
        Returns:
        - dL_dInput: Gradient with respect to the input of the convolutional layer.
        """
        dL_dInput = np.zeros_like(self.input, dtype=np.float64)
        dL_dFilters = np.zeros_like(self.filters, dtype=np.float64)

        # Compute gradient with respect to filters
        for f in range(self.number_of_filter):
            for d in range(self.input.shape[0]):
                for i in range(0, self.input_size - self.filter_size + 1):
                    for j in range(0, self.input_size - self.filter_size + 1):
                        receptive_field = self.input[d, i:i+self.filter_size, j:j+self.filter_size]
                        dL_dFilters[f] += dL_dOut[f, i, j] * receptive_field

        # Compute gradient with respect to biases
        dL_dBiases = np.sum(dL_dOut, axis=(0, 1))

        # Compute gradient with respect to input
        for f in range(self.number_of_filter):
            for d in range(self.input.shape[0]):
                for i in range(0, self.input_size - self.filter_size + 1, self.stride_size):
                    for j in range(0, self.input_size - self.filter_size + 1, self.stride_size):
                        dL_dInput[d, i:i+self.filter_size, j:j+self.filter_size] += dL_dOut[f, i//self.stride_size, j//self.stride_size] * self.filters[f]

        
        # Store gradients for updating weights and biases
        self.delta_filters = dL_dFilters
        self.delta_biases = dL_dBiases

        return dL_dInput
