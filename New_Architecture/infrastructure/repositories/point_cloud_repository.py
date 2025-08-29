# domain/repositories/point_cloud_repository.py
"""Repository-Interfaces für Datenzugriff"""

from abc import ABC, abstractmethod
from typing import Protocol, Optional, Tuple, Dict, Any
import numpy as np
from pathlib import Path


class PointCloudRepository(Protocol):
    """Interface für Punktwolken-Datenzugriff"""
    
    def load_point_cloud(self, path: str) -> np.ndarray:
        """Lädt eine Punktwolke von einem Pfad"""
        ...
    
    def save_point_cloud(self, data: np.ndarray, path: str) -> None:
        """Speichert eine Punktwolke"""
        ...
    
    def load_distances(self, path: str) -> np.ndarray:
        """Lädt M3C2-Distanzen"""
        ...
    
    def save_distances(self, distances: np.ndarray, path: str) -> None:
        """Speichert M3C2-Distanzen"""
        ...
    
    def save_distances_with_coordinates(
        self, 
        coordinates: np.ndarray,
        distances: np.ndarray, 
        path: str
    ) -> None:
        """Speichert Distanzen mit Koordinaten"""
        ...
    
    def exists(self, path: str) -> bool:
        """Prüft ob Datei existiert"""
        ...


class ParameterRepository(Protocol):
    """Interface für Parameter-Persistierung"""
    
    def load_params(self, path: str) -> Dict[str, float]:
        """Lädt M3C2-Parameter"""
        ...
    
    def save_params(self, params: Dict[str, float], path: str) -> None:
        """Speichert M3C2-Parameter"""
        ...


class StatisticsRepository(Protocol):
    """Interface für Statistik-Export"""
    
    def save_statistics(
        self, 
        statistics: Dict[str, Any],
        path: str,
        format: str = "excel"
    ) -> None:
        """Speichert berechnete Statistiken"""
        ...
    
    def append_statistics(
        self,
        statistics: Dict[str, Any],
        path: str,
        sheet_name: str = "Results"
    ) -> None:
        """Hängt Statistiken an bestehende Datei an"""
        ...