import numpy as np

class DetectorStage: 
    def __init__(
        self,
        feature_maps: np.ndarray,
        processed_feature_maps: np.ndarray = []
    ) -> None:
        self.feature_maps = feature_maps
        self.processed_feature_maps = processed_feature_maps

    def setFeatureMaps(self, feature_maps):
        self.feature_maps = feature_maps

    def activation_function(self, feature):
        return np.maximum(0, feature)
    
    def detect(self):
        self.processed_feature_maps = np.array([self.activation_function(feature) for feature in self.feature_maps])
