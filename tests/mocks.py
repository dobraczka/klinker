import numpy as np
from typing import Dict, Optional


class MockKeyedVector:
    dimension: int = 3
    embeddings: Optional[Dict[str, np.ndarray]] = None

    def __init__(self, dimension: int = 3, embeddings: Optional[Dict[str, np.ndarray]] = None):
        self.dimension = dimension
        self.embeddings = embeddings

    def __getitem__(self, key: str):
        if self.embeddings is None:
            return np.random.rand(self.dimension)
        else:
            self.embeddings[key]

    @classmethod
    def load(cls, path, mmap):
        return MockKeyedVector(dimension=cls.dimension, embeddings=cls.embeddings)


class MockGensimDownloader:
    def __init__(self, dimension: int = 3):
        self.dimension = dimension

    def load(self, name: str):
        return MockKeyedVector(dimension=self.dimension)
