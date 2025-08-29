# New_Architecture/infrastructure/repositories/distance_repository.py
"""Repository für M3C2 Distanzen und verwandte Daten"""

import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DistanceRepository:
    """
    Repository für M3C2 Distanzen mit Support für:
    - Einfache Distanz-Arrays
    - Distanzen mit Koordinaten
    - Outlier/Inlier getrennte Dateien
    - CloudCompare M3C2 Ausgaben
    """
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def load_distances(self, path: str) -> np.ndarray:
        """
        Lädt M3C2 Distanzen aus Datei.
        
        Unterstützt:
        - Einfache 1-Spalten Textdateien
        - CloudCompare CSV mit Semikolon
        - Dateien mit Header
        """
        full_path = self._resolve_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"Distance file not found: {full_path}")
        
        logger.info(f"Loading distances from {full_path}")
        
        # Erkenne Format
        if self._is_cloudcompare_format(full_path):
            return self._load_cloudcompare_distances(full_path)
        else:
            return self._load_simple_distances(full_path)
    
    def load_distances_with_coordinates(
        self, 
        path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lädt Distanzen mit Koordinaten.
        
        Returns:
            Tuple von (coordinates, distances)
        """
        full_path = self._resolve_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")
        
        # Prüfe ob Header vorhanden
        has_header = self._has_header(full_path)
        
        # Lade Daten
        data = np.loadtxt(
            str(full_path),
            skiprows=1 if has_header else 0,
            ndmin=2
        )
        
        if data.shape[1] < 4:
            raise ValueError(f"Expected at least 4 columns (x,y,z,distance), got {data.shape[1]}")
        
        coordinates = data[:, :3]
        distances = data[:, 3]
        
        logger.info(f"Loaded {len(distances)} distances with coordinates")
        
        return coordinates, distances
    
    def save_distances(
        self,
        distances: np.ndarray,
        path: str,
        fmt: str = "%.6f"
    ) -> None:
        """Speichert Distanz-Array"""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        nan_percentage = np.isnan(distances).sum() / len(distances) * 100
        logger.info(
            f"Saving {len(distances)} distances to {full_path} "
            f"({nan_percentage:.1f}% NaN)"
        )
        
        np.savetxt(str(full_path), distances, fmt=fmt)
    
    def save_distances_with_coordinates(
        self,
        coordinates: np.ndarray,
        distances: np.ndarray,
        path: str,
        header: Optional[str] = "x y z distance",
        fmt: str = "%.6f"
    ) -> None:
        """Speichert Distanzen mit Koordinaten"""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        if coordinates.shape[0] != distances.shape[0]:
            raise ValueError(
                f"Shape mismatch: coordinates {coordinates.shape} "
                f"vs distances {distances.shape}"
            )
        
        # Kombiniere Daten
        combined = np.column_stack((coordinates, distances))
        
        logger.info(f"Saving coordinates with distances to {full_path}")
        
        np.savetxt(
            str(full_path),
            combined,
            fmt=fmt,
            header=header if header else "",
            comments='#' if header else ''
        )
    
    def split_by_outliers(
        self,
        coordinates: np.ndarray,
        distances: np.ndarray,
        outlier_mask: np.ndarray,
        base_path: str,
        method: str = "rmse"
    ) -> Tuple[str, str]:
        """
        Teilt Daten in Outlier und Inlier auf und speichert sie.
        
        Returns:
            Tuple von (outlier_path, inlier_path)
        """
        base = Path(base_path)
        
        # Erstelle Dateinamen
        outlier_path = base.parent / f"{base.stem}_outlier_{method}.txt"
        inlier_path = base.parent / f"{base.stem}_inlier_{method}.txt"
        
        # Filtere Daten
        outlier_coords = coordinates[outlier_mask]
        outlier_dists = distances[outlier_mask]
        
        inlier_mask = ~outlier_mask & ~np.isnan(distances)
        inlier_coords = coordinates[inlier_mask]
        inlier_dists = distances[inlier_mask]
        
        # Speichere Outliers
        if len(outlier_coords) > 0:
            self.save_distances_with_coordinates(
                outlier_coords,
                outlier_dists,
                str(outlier_path)
            )
            logger.info(f"Saved {len(outlier_coords)} outliers to {outlier_path}")
        
        # Speichere Inliers
        if len(inlier_coords) > 0:
            self.save_distances_with_coordinates(
                inlier_coords,
                inlier_dists,
                str(inlier_path)
            )
            logger.info(f"Saved {len(inlier_coords)} inliers to {inlier_path}")
        
        return str(outlier_path), str(inlier_path)
    
    def load_cloudcompare_m3c2(
        self,
        path: str
    ) -> Dict[str, np.ndarray]:
        """
        Lädt CloudCompare M3C2 Ausgabe mit allen Spalten.
        
        Returns:
            Dictionary mit allen gefundenen Spalten
        """
        full_path = self._resolve_path(path)
        
        # Lade als DataFrame
        df = pd.read_csv(full_path, sep=';')
        
        result = {}
        
        # Standard M3C2 Spalten
        column_mapping = {
            'M3C2 distance': 'distances',
            'distance uncertainty': 'uncertainties',
            'STD cloud1': 'std_cloud1',
            'STD cloud2': 'std_cloud2',
            'Npoints cloud1': 'npoints_cloud1',
            'Npoints cloud2': 'npoints_cloud2',
            'X': 'x',
            'Y': 'y',
            'Z': 'z',
            'Nx': 'nx',
            'Ny': 'ny',
            'Nz': 'nz'
        }
        
        for cc_name, key in column_mapping.items():
            if cc_name in df.columns:
                result[key] = df[cc_name].values
        
        logger.info(
            f"Loaded CloudCompare M3C2 data with columns: {list(result.keys())}"
        )
        
        return result
    
    def merge_distance_files(
        self,
        file_paths: List[str],
        output_path: str
    ) -> None:
        """
        Kombiniert mehrere Distanz-Dateien.
        
        Nützlich für Batch-Verarbeitung oder Teil-Ergebnisse.
        """
        all_distances = []
        
        for path in file_paths:
            distances = self.load_distances(path)
            all_distances.append(distances)
        
        # Kombiniere
        combined = np.concatenate(all_distances)
        
        # Speichere
        self.save_distances(combined, output_path)
        
        logger.info(
            f"Merged {len(file_paths)} files into {output_path} "
            f"({len(combined)} total distances)"
        )
    
    # ============= Private Hilfsmethoden =============
    
    def _resolve_path(self, path: str) -> Path:
        """Löst Pfad relativ zum base_path auf"""
        p = Path(path)
        if p.is_absolute():
            return p
        
        # Versuche verschiedene Locations
        # 1. Direkt im base_path
        test_path = self.base_path / path
        if test_path.exists():
            return test_path
        
        # 2. In data/ Unterordner
        test_path = self.base_path / "data" / path
        if test_path.exists():
            return test_path
        
        # Default: relativ zu base_path
        return self.base_path / path
    
    def _is_cloudcompare_format(self, path: Path) -> bool:
        """Prüft ob Datei CloudCompare Format ist"""
        with open(path, 'r') as f:
            first_line = f.readline()
            # CloudCompare verwendet Semikolon als Separator
            return ';' in first_line and 'M3C2' in first_line
    
    def _has_header(self, path: Path) -> bool:
        """Prüft ob Datei einen Header hat"""
        with open(path, 'r') as f:
            first_line = f.readline().strip()
            
            # Prüfe ob erste Zeile numerisch ist
            parts = first_line.split()
            if not parts:
                return False
            
            try:
                # Versuche als Zahlen zu parsen
                [float(p) for p in parts]
                return False  # Kein Header, direkt Daten
            except ValueError:
                return True  # Header vorhanden
    
    def _load_simple_distances(self, path: Path) -> np.ndarray:
        """Lädt einfache Distanz-Datei (1 Spalte)"""
        # Prüfe auf Header
        has_header = self._has_header(path)
        
        data = np.loadtxt(
            str(path),
            skiprows=1 if has_header else 0,
            ndmin=1
        )
        
        # Stelle sicher dass es 1D ist
        if data.ndim > 1:
            if data.shape[1] == 1:
                data = data.flatten()
            else:
                # Nimm letzte Spalte als Distanzen
                data = data[:, -1]
        
        return data
    
    def _load_cloudcompare_distances(self, path: Path) -> np.ndarray:
        """Lädt CloudCompare M3C2 Distanzen"""
        df = pd.read_csv(path, sep=';')
        
        # Suche nach Distanz-Spalte
        distance_columns = ['M3C2 distance', 'M3C2_distance', 'distance', 'Distance']
        
        for col in distance_columns:
            if col in df.columns:
                return df[col].values.astype(np.float64)
        
        # Fallback: Erste numerische Spalte
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            logger.warning(
                f"No standard distance column found, using {numeric_cols[0]}"
            )
            return df[numeric_cols[0]].values.astype(np.float64)
        
        raise ValueError(f"No distance column found in CloudCompare file: {path}")