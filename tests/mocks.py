import numpy as np
import torch
from klinker.typing import GeneralVectorLiteral

class MockKeyedVector:
    def __init__(self, dimension: int = 3):
        self.dimension = dimension

    def __getitem__(self, key: str):
        return np.random.rand(self.dimension)

class MockGensimDownloader:
    def __init__(self, dimension: int = 3):
        self.dimension = dimension

    def load(self, name: str):
        return MockKeyedVector(dimension=self.dimension)
