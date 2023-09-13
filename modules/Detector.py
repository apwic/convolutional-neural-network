import numpy as np

class DetectorStage: 
    def __init__(
        self,
        inputs: np.ndarray = None,
    ) -> None:
        self.inputs = inputs
        self.feature_maps = []

    def __str__(self) -> str:
        return f"\nDETECTOR STAGE\n--------\nInput: {self.inputs}\n\nOutput: {self.feature_maps}\n"

    def setInput(self, inputs):
        self.inputs = inputs

    def getOutput(self):
        return self.feature_maps

    def activation_function(self, feature):
        return np.maximum(0, feature)
    
    def calculate(self):
        for feature_map in self.inputs:
            self.feature_maps.append(np.array([self.activation_function(feature) for feature in feature_map]))
        self.feature_maps = np.array(self.feature_maps)
