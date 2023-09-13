import numpy as np

class FlattenLayer:
    def __init__(
        self,
        input_size : int
    ) -> None:
        self.input_size = input_size
        self.inputs: np.ndarray = None

        self.output_size = input_size * input_size
        self.ouputs: np.ndarray = None
    
    def setInput(self, inputs: np.ndarray):
        self.inputs = inputs

    def flatten(self):
        flattened_inputs = []
        for feature_map in self.inputs:
            for i in range(0, self.input_size):
                flattened_inputs.append(feature_map[i])
                
        self.ouputs = np.concatenate(flattened_inputs)