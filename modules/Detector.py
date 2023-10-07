import numpy as np

class DetectorStage: 
    def __init__(
        self,
        input: np.ndarray = None,
    ) -> None:
        self.input = input
        self.feature_maps = []

    def __str__(self) -> str:
        return f"\nDETECTOR STAGE\n--------\nInput: {self.input}\n\nOutput: {self.feature_maps}\n"

    def setInput(self, input):
        self.input = input

    def getOutput(self):
        return self.feature_maps

    def activation_function(self, feature, deriv=False):
        if (deriv):
            return np.where(feature <= 0, 0, 1)
        
        return np.maximum(0, feature)
    
    def calculate(self):
        for feature_map in self.input:
            if (type(self.feature_maps) is np.ndarray):
                self.feature_maps = self.feature_maps.tolist()
            self.feature_maps.append(np.array([self.activation_function(feature) for feature in feature_map]))
        self.feature_maps = np.array(self.feature_maps)

    def backprop(self, dL_dOut):
        """
        Backpropagation for the detector stage.
        
        Parameters:
        - dL_dOut: Gradient with respect to the output of the detector stage.
        
        Returns:
        - dL_dInput: Gradient with respect to the input of the detector stage.
        """
        dL_dInput = dL_dOut * self.activation_function(self.input, deriv=True)
        return dL_dInput

    def resetOutput(self):
        self.feature_maps = []