import numpy as np

class FlattenLayer:
    def __init__(
        self,
        input_size : int
    ) -> None:
        self.input_size = input_size
        self.inputs: np.ndarray = None

        self.output_size = input_size * input_size
        self.outputs: np.ndarray = None

    def __str__(self):
        return f"\nFLATTEN LAYER\n--------\nInput: {self.inputs}\n\nOutput: {self.outputs}\n"
    
    def setInput(self, inputs: np.ndarray):
        self.inputs = inputs

    def flatten(self):
        flattened_inputs = []
        for feature_map in self.inputs:
            for i in range(0, self.input_size):
                flattened_inputs.append(feature_map[i])
                
        self.outputs = np.concatenate(flattened_inputs)