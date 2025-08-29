# infrastructure/repositories/file_point_cloud_repository.py
"""Konkrete Repository-Implementierungen für Dateisystem"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import py4dgeo

logger = logging.getLogger(__name__)


class FilePointCloudRepository:
    """Dateisystem-basierte Implementierung des PointCloudRepository"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def load_point_cloud(self, path: str) -> py4dgeo.Epoch:
        """Lädt eine Punktwolke als py4dgeo.Epoch"""
        full_path = self._resolve_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"Point cloud not found: {full_path}")
        
        logger.info(f"Loading point cloud from {full_path}")
        
        if full_path.suffix.lower() == '.ply':
            return py4dgeo.read_from_ply(str(full_path))
        elif full_path.suffix.lower() in ['.txt', '.xyz']:
            return py4dgeo.Epoch(np.loadtxt(str(full_path)))
        else:
            raise ValueError(f"Unsupported file format: {full_path.suffix}")
    
    def save_point_cloud(self, data: np.ndarray, path: str) -> None:
        """Speichert eine Punktwolke"""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving point cloud to {full_path}")
        
        if isinstance(data, py4dgeo.Epoch):
            data = np.asarray(data.cloud)
        
        np.savetxt(str(full_path), data, fmt="%.6f")
    
    def load_distances(self, path: str) -> np.ndarray:
        """Lädt M3C2-Distanzen"""
        full_path = self._resolve_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"Distances file not found: {full_path}")
        
        logger.info(f"Loading distances from {full_path}")
        return np.loadtxt(str(full_path))
    
    def save_distances(self, distances: np.ndarray, path: str) -> None:
        """Speichert M3C2-Distanzen"""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        nan_percentage = np.isnan(distances).sum() / len(distances) * 100
        logger.info(f"Saving distances to {full_path} ({len(distances)} values, {nan_percentage:.1f}% NaN)")
        
        np.savetxt(str(full_path), distances, fmt="%.6f")
    
    def save_distances_with_coordinates(
        self, 
        coordinates: np.ndarray,
        distances: np.ndarray, 
        path: str
    ) -> None:
        """Speichert Distanzen mit XYZ-Koordinaten"""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        if coordinates.shape[0] != distances.shape[0]:
            raise ValueError(f"Shape mismatch: coordinates {coordinates.shape} vs distances {distances.shape}")
        
        combined = np.column_stack((coordinates, distances))
        header = "x y z distance"
        
        logger.info(f"Saving coordinates with distances to {full_path}")
        np.savetxt(str(full_path), combined, fmt="%.6f", header=header)
    
    def exists(self, path: str) -> bool:
        """Prüft ob Datei existiert"""
        full_path = self._resolve_path(path)
        return full_path.exists()
    
    def _resolve_path(self, path: str) -> Path:
        """Löst relativen Pfad zu absolutem Pfad auf"""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_path / p


class FileParameterRepository:
    """Dateisystem-basierte Parameter-Persistierung"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        
    def load_params(self, path: str) -> Dict[str, float]:
        """Lädt M3C2-Parameter aus Textdatei"""
        full_path = self._resolve_path(path)
        
        if not full_path.exists():
            logger.warning(f"Parameter file not found: {full_path}")
            return {}
        
        params = {}
        with open(full_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=')
                    try:
                        params[key] = float(value)
                    except ValueError:
                        logger.warning(f"Could not parse parameter: {line.strip()}")
        
        logger.info(f"Loaded parameters from {full_path}: {params}")
        return params
    
    def save_params(self, params: Dict[str, float], path: str) -> None:
        """Speichert M3C2-Parameter in Textdatei"""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w') as f:
            for key, value in params.items():
                f.write(f"{key}={value}\n")
        
        logger.info(f"Saved parameters to {full_path}: {params}")
    
    def _resolve_path(self, path: str) -> Path:
        """Löst relativen Pfad zu absolutem Pfad auf"""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_path / p


class FileStatisticsRepository:
    """Dateisystem-basierte Statistik-Persistierung"""
    
    def __init__(self, output_path: str = "outputs"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def save_statistics(
        self, 
        statistics: pd.DataFrame,
        path: str,
        format: str = "excel"
    ) -> None:
        """Speichert Statistiken in gewünschtem Format"""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "excel":
            statistics.to_excel(full_path, index=False)
            logger.info(f"Saved statistics to Excel: {full_path}")
        elif format == "json":
            statistics.to_json(full_path, orient="records", indent=2)
            logger.info(f"Saved statistics to JSON: {full_path}")
        elif format == "csv":
            statistics.to_csv(full_path, index=False)
            logger.info(f"Saved statistics to CSV: {full_path}")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def append_statistics(
        self,
        statistics: pd.DataFrame,
        path: str,
        sheet_name: str = "Results"
    ) -> None:
        """Hängt Statistiken an bestehende Excel-Datei an"""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        if full_path.exists() and full_path.suffix == '.xlsx':
            # Lese existierende Daten
            with pd.ExcelFile(full_path) as xls:
                if sheet_name in xls.sheet_names:
                    existing = pd.read_excel(xls, sheet_name=sheet_name)
                    combined = pd.concat([existing, statistics], ignore_index=True)
                else:
                    combined = statistics
            
            # Schreibe alle Sheets zurück
            with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
                for sheet in xls.sheet_names:
                    if sheet != sheet_name:
                        pd.read_excel(xls, sheet_name=sheet).to_excel(
                            writer, sheet_name=sheet, index=False
                        )
                combined.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # Neue Datei erstellen
            statistics.to_excel(full_path, sheet_name=sheet_name, index=False)
        
        logger.info(f"Appended statistics to {full_path} (sheet: {sheet_name})")
    
    def _resolve_path(self, path: str) -> Path:
        """Löst relativen Pfad zu absolutem Pfad auf"""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.output_path / p