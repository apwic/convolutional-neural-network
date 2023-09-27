import numpy as np

class FlattenLayer:
    def __init__(
        self,
        input_size : int
    ) -> None:
        self.input_size = input_size
        self.input: np.ndarray = None

        self.output_size = input_size * input_size
        self.output: np.ndarray = None

    def __str__(self):
        return f"\nFLATTEN LAYER\n--------\nInput: {self.input}\n\nOutput: {self.output}\n"
    
    def setInput(self, input: np.ndarray):
        self.input = input

    def flatten(self):
        flattened_inputs = []
        for feature_map in self.input:
            for i in range(0, self.input_size):
                flattened_inputs.append(feature_map[i])
                
        self.output = np.concatenate(flattened_inputs)