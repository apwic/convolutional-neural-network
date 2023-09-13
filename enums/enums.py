from enum import Enum

class ActivationFunction(Enum):
    RELU = 0
    SIGMOID = 1

class PoolingMode(Enum):
    POOLING_MAX = 0
    POOLING_AVG = 1