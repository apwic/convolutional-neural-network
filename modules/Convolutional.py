import numpy as np

class ConvolutionalStage: 
    def __init__(
        self,
        num_of_input: int,
        input_size: int,
        filter_size: int,
        number_of_filter: int,
        padding_size: int = 0,
        stride_size: int = 1,
        learning_rate: float = 0.01
    ) -> None:
        self.num_of_input = num_of_input
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
    
        self.learning_rate = learning_rate

    def __str__(self) -> str:
        return f"\nCONVOLUTION STAGE\n--------\nInput: {self.input}\n\nOutput: {self.feature_maps}\n"

    def setInput(self, input: np.ndarray) :
        self.num_of_input = input.shape[0]
        self.input_size = input[0].shape[0]
        self.input = input
        
        if (self.padding_size > 0):
            self.addPadding()
        
        self.feature_map_size = (self.input_size - self.filter_size + 2 * self.padding_size) // self.stride_size + 1

    def setParams(self, weights = np.ndarray):
        self.filters = weights.astype(np.float64)

    def setBiases(self, biases = np.ndarray):
        self.biases = biases

    def resetParams(self):
        self.filters = np.random.randn(self.number_of_filter, self.filter_size, self.filter_size) * np.sqrt(2. / self.filter_size * self.filter_size)

    def getOutput(self):
        return self.feature_maps
    
    def getOutputShape(self):
        return (self.number_of_filter, self.feature_map_size)
    
    def getParamsCount(self):
        return self.number_of_filter * ((self.filter_size * self.filter_size * self.num_of_input) + 1)

    def addPadding(self):
        num_channels = self.input.shape[0]
        padded_inputs = np.zeros((num_channels, self.input_size + 2 * self.padding_size, self.input_size + 2 * self.padding_size), dtype=float)

        for c in range(num_channels):
            for i in range(self.input_size):
                for j in range(self.input_size):
                    padded_inputs[c, i + self.padding_size, j + self.padding_size] = self.input[c, i, j]
            
        self.input = padded_inputs

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
            for d in range(self.num_of_input):
                for i in range(0, self.input_size - self.filter_size + 1, self.stride_size):
                    for j in range(0, self.input_size - self.filter_size + 1, self.stride_size):
                        receptive_field = self.input[d, i:i+self.filter_size, j:j+self.filter_size]
                        dL_dFilters[f] += dL_dOut[f, i//self.stride_size, j//self.stride_size] * receptive_field

        # Compute gradient with respect to biases
        dL_dBiases = np.sum(dL_dOut, axis=(1, 2))

        # Compute gradient with respect to input
        for f in range(self.number_of_filter):
            for d in range(self.num_of_input):
                for i in range(0, self.input_size - self.filter_size + 1, self.stride_size):
                    for j in range(0, self.input_size - self.filter_size + 1, self.stride_size):
                        dL_dInput[d, i:i+self.filter_size, j:j+self.filter_size] += dL_dOut[f, i//self.stride_size, j//self.stride_size] * self.filters[f]


        # Store gradients for updating weights and biases
        self.delta_filters = dL_dFilters.astype(np.float64)
        self.delta_biases = dL_dBiases.astype(np.float64)

        return dL_dInput
    
    def update_weights_and_biases(self):
        self.filters -= self.learning_rate * self.delta_filters.astype(np.float64)
        self.biases -= self.learning_rate * self.delta_biases.astype(np.float64)

    def resetOutput(self):
        self.feature_map: np.ndarray = []

    def resetAll(self):
        self.feature_maps: np.ndarray = []
        # backprop attribute
        self.delta_filters = None
        self.delta_biases = None
