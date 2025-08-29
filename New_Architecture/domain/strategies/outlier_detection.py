# domain/strategies/outlier_detection.py
from abc import ABC, abstractmethod
import numpy as np

class OutlierDetectionStrategy(ABC):
    """Basis für Outlier-Detection Strategien"""
    
    @abstractmethod
    def detect(self, distances: np.ndarray) -> np.ndarray:
        """Gibt Boolean-Array zurück (True = Outlier)"""
        pass

class RMSEOutlierStrategy(OutlierDetectionStrategy):
    def __init__(self, multiplier: float = 3.0):
        self.multiplier = multiplier
    
    def detect(self, distances: np.ndarray) -> np.ndarray:
        valid = distances[~np.isnan(distances)]
        rms = np.sqrt(np.mean(valid ** 2))
        threshold = self.multiplier * rms
        return np.abs(distances) > threshold

class MADOutlierStrategy(OutlierDetectionStrategy):
    def __init__(self, multiplier: float = 3.0):
        self.multiplier = multiplier
    
    def detect(self, distances: np.ndarray) -> np.ndarray:
        valid = distances[~np.isnan(distances)]
        median = np.median(valid)
        mad = np.median(np.abs(valid - median))
        threshold = self.multiplier * 1.4826 * mad
        return np.abs(distances - median) > threshold