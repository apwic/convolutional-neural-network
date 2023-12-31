import numpy as np

class FlattenLayer:
    def __init__(
        self,
        num_of_input: int,
        input_size : int
    ) -> None:
        self.num_of_input = num_of_input
        self.input_size = input_size
        self.input: np.ndarray = None
        self.output: np.ndarray = None

    def __str__(self):
        return f"\nFLATTEN LAYER\n--------\nInput: {self.input}\n\nOutput: {self.output}\n"
    
    def setInput(self, input: np.ndarray):
        self.input = input

    def getOutput(self):
        return self.output
    
    def getOutputShape(self):
        return self.num_of_input * self.input_size * self.input_size
    
    def getData(self):
        return {
            "type": "flatten",
            "params": {}
        }

    def flatten(self):
        flattened_inputs = []
        for feature_map in self.input:
            for i in range(0, self.input_size):
                flattened_inputs.append(feature_map[i])
                
        self.output = np.concatenate(flattened_inputs)

    def backprop(self, dL_dOut):
        dL_df = dL_dOut.reshape(self.input.shape) 

        return dL_df
    
    def resetOutput(self):
        self.output = None
