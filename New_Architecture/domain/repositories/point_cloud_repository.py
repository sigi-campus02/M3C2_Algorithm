# domain/repositories/point_cloud_repository.py
from abc import ABC, abstractmethod
from typing import Protocol
import numpy as np

class PointCloudRepository(Protocol):
    """Interface für Punktwolken-Datenzugriff"""
    
    def load_point_cloud(self, path: str) -> np.ndarray:
        ...
    
    def save_point_cloud(self, data: np.ndarray, path: str) -> None:
        ...
    
    def save_distances(self, distances: np.ndarray, path: str) -> None:
        ...
