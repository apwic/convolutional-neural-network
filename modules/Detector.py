import numpy as np

class DetectorStage: 
    def __init__(
        self,
        input: np.ndarray,
    ) -> None:
        self.input = input
        self.feature_maps = []

    def setInput(self, input):
        self.input = input

    def activation_function(self, feature):
        return np.maximum(0, feature)
    
    def calculate(self):
        for feature_map in self.input:
            self.feature_maps.append(np.array([self.activation_function(feature) for feature in feature_map]))
        self.feature_maps = np.array(self.feature_maps)
