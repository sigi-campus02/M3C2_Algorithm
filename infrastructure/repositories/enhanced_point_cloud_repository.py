# New_Architecture/infrastructure/repositories/enhanced_point_cloud_repository.py
"""Erweiterte Repository-Implementierung mit Multi-Format Support"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import py4dgeo

# Optional imports
try:
    from plyfile import PlyData, PlyElement
    PLYFILE_AVAILABLE = True
except ImportError:
    PLYFILE_AVAILABLE = False

try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedPointCloudRepository:
    """
    Erweiterte Repository-Implementierung mit Support für:
    - XYZ (Standard ASCII)
    - PLY (Binary/ASCII)
    - LAS/LAZ (LiDAR)
    - OBJ (Wavefront)
    - GPC (Custom format)
    - CloudCompare Ausgaben
    """
    
    SUPPORTED_FORMATS = {'.xyz', '.ply', '.las', '.laz', '.obj', '.gpc', '.txt'}
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._format_handlers = self._initialize_handlers()
        
    def _initialize_handlers(self) -> Dict[str, callable]:
        """Initialisiert Format-spezifische Handler"""
        return {
            '.xyz': self._load_xyz,
            '.txt': self._load_txt,  # Kann XYZ oder CloudCompare sein
            '.ply': self._load_ply,
            '.las': self._load_las,
            '.laz': self._load_las,  # Gleicher Handler
            '.obj': self._load_obj,
            '.gpc': self._load_gpc,
        }
    
    def detect_format(self, path: Union[str, Path]) -> str:
        """Erkennt das Format einer Datei"""
        path = Path(path)
        
        # Zuerst nach Dateiendung
        suffix = path.suffix.lower()
        if suffix in self.SUPPORTED_FORMATS:
            return suffix
        
        # Bei .txt müssen wir den Inhalt prüfen
        if suffix == '.txt':
            return self._detect_txt_format(path)
        
        raise ValueError(f"Unsupported format: {suffix}")
    
    def load_point_cloud(self, path: str, format: Optional[str] = None) -> py4dgeo.Epoch:
        """
        Lädt eine Punktwolke in beliebigem Format.
        
        Args:
            path: Pfad zur Datei
            format: Explizites Format (optional, wird sonst erkannt)
            
        Returns:
            py4dgeo.Epoch Objekt
        """
        full_path = self._resolve_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"Point cloud not found: {full_path}")
        
        # Format erkennen wenn nicht angegeben
        if format is None:
            format = self.detect_format(full_path)
        
        logger.info(f"Loading point cloud from {full_path} (format: {format})")
        
        # Handler aufrufen
        handler = self._format_handlers.get(format)
        if handler is None:
            raise ValueError(f"No handler for format: {format}")
        
        points = handler(full_path)
        
        # Konvertiere zu py4dgeo.Epoch
        if isinstance(points, py4dgeo.Epoch):
            return points
        else:
            return py4dgeo.Epoch(points)
    
    def save_point_cloud(
        self, 
        data: Union[np.ndarray, py4dgeo.Epoch],
        path: str,
        format: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Speichert eine Punktwolke in gewünschtem Format.
        
        Args:
            data: Punktwolken-Daten
            path: Ziel-Pfad
            format: Ausgabeformat (wird aus Pfad erkannt wenn None)
            **kwargs: Format-spezifische Optionen
        """
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format aus Dateiendung wenn nicht angegeben
        if format is None:
            format = full_path.suffix.lower()
        
        # Konvertiere zu numpy array wenn nötig
        if isinstance(data, py4dgeo.Epoch):
            points = np.asarray(data.cloud)
        else:
            points = data
        
        logger.info(f"Saving {len(points)} points to {full_path} (format: {format})")
        
        if format in ['.xyz', '.txt']:
            self._save_xyz(points, full_path, **kwargs)
        elif format == '.ply':
            self._save_ply(points, full_path, **kwargs)
        elif format in ['.las', '.laz']:
            self._save_las(points, full_path, format, **kwargs)
        else:
            raise ValueError(f"Unsupported save format: {format}")
    
    # ============= Format-spezifische Loader =============
    
    def _load_xyz(self, path: Path) -> np.ndarray:
        """Lädt XYZ ASCII Format"""
        return np.loadtxt(str(path), dtype=np.float64)
    
    def _load_txt(self, path: Path) -> np.ndarray:
        """Lädt TXT (kann XYZ oder CloudCompare sein)"""
        # Prüfe ob CloudCompare Format
        if self._is_cloudcompare_format(path):
            return self._load_cloudcompare(path)
        else:
            return self._load_xyz(path)
    
    def _load_ply(self, path: Path) -> np.ndarray:
        """Lädt PLY Format"""
        if not PLYFILE_AVAILABLE:
            raise RuntimeError("PLY support requires 'plyfile' package. Install with: pip install plyfile")
        
        plydata = PlyData.read(str(path))
        vertex = plydata['vertex']
        
        # Extrahiere XYZ Koordinaten
        x = vertex['x']
        y = vertex['y'] 
        z = vertex['z']
        
        points = np.vstack([x, y, z]).T.astype(np.float64)
        
        # Optional: Zusätzliche Attribute wenn vorhanden
        if 'nx' in vertex.data.dtype.names:
            # Normale vorhanden
            nx = vertex['nx']
            ny = vertex['ny']
            nz = vertex['nz']
            normals = np.vstack([nx, ny, nz]).T
            points = np.hstack([points, normals])
        
        return points
    
    def _load_las(self, path: Path) -> np.ndarray:
        """Lädt LAS/LAZ Format"""
        if not LASPY_AVAILABLE:
            raise RuntimeError("LAS/LAZ support requires 'laspy' package. Install with: pip install laspy")
        
        try:
            las = laspy.read(str(path))
        except Exception as e:
            if path.suffix == '.laz':
                raise RuntimeError("LAZ files require additional backend. Install with: pip install 'laspy[lazrs]'") from e
            raise
        
        # Skaliere Koordinaten
        x = las.x
        y = las.y
        z = las.z
        
        points = np.vstack([x, y, z]).T.astype(np.float64)
        
        # Optional: Zusätzliche Attribute
        if hasattr(las, 'intensity'):
            intensity = las.intensity.reshape(-1, 1)
            points = np.hstack([points, intensity])
        
        return points
    
    def _load_obj(self, path: Path) -> np.ndarray:
        """Lädt Wavefront OBJ Format (nur Vertices)"""
        vertices = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):  # Vertex line
                    parts = line.split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        vertices.append([x, y, z])
        
        if not vertices:
            raise ValueError(f"No vertices found in OBJ file: {path}")
        
        return np.array(vertices, dtype=np.float64)
    
    def _load_gpc(self, path: Path) -> np.ndarray:
        """Lädt GPC Format (custom whitespace-separated)"""
        return np.loadtxt(str(path), dtype=np.float64, usecols=(0, 1, 2))
    
    def _load_cloudcompare(self, path: Path) -> np.ndarray:
        """Lädt CloudCompare M3C2 Ausgabe"""
        # CloudCompare kann verschiedene Formate haben
        # Häufigstes: Semikolon-getrennte CSV mit Header
        
        try:
            # Versuche als CSV zu laden
            df = pd.read_csv(path, sep=';')
            
            # Suche nach Koordinaten-Spalten
            coord_cols = []
            for col_name in ['X', 'Y', 'Z', 'x', 'y', 'z']:
                if col_name in df.columns:
                    coord_cols.append(col_name)
            
            if len(coord_cols) >= 3:
                return df[coord_cols[:3]].values.astype(np.float64)
            
            # Fallback: Nimm erste 3 numerische Spalten
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 3:
                return df[numeric_cols[:3]].values.astype(np.float64)
            
        except Exception:
            pass
        
        # Fallback: Behandle als normale XYZ
        return self._load_xyz(path)
    
    # ============= Format-spezifische Saver =============
    
    def _save_xyz(self, points: np.ndarray, path: Path, **kwargs) -> None:
        """Speichert als XYZ ASCII"""
        fmt = kwargs.get('fmt', '%.6f')
        header = kwargs.get('header', '')
        
        np.savetxt(str(path), points, fmt=fmt, header=header)
    
    def _save_ply(self, points: np.ndarray, path: Path, **kwargs) -> None:
        """Speichert als PLY"""
        if not PLYFILE_AVAILABLE:
            raise RuntimeError("PLY support requires 'plyfile' package")
        
        # Erstelle Vertex-Array
        vertex = np.array(
            [(p[0], p[1], p[2]) for p in points],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        )
        
        # Optional: Füge Farben hinzu wenn vorhanden
        if 'colors' in kwargs:
            colors = kwargs['colors']
            vertex = np.array(
                [(p[0], p[1], p[2], c[0], c[1], c[2]) 
                 for p, c in zip(points, colors)],
                dtype=[
                    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
                ]
            )
        
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el]).write(str(path))
    
    def _save_las(self, points: np.ndarray, path: Path, format: str, **kwargs) -> None:
        """Speichert als LAS/LAZ"""
        if not LASPY_AVAILABLE:
            raise RuntimeError("LAS/LAZ support requires 'laspy' package")
        
        # Erstelle neues LAS-File
        header = laspy.LasHeader(point_format=3, version="1.2")
        las = laspy.LasData(header)
        
        # Setze Koordinaten
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]
        
        # Optional: Zusätzliche Attribute
        if points.shape[1] > 3:
            las.intensity = points[:, 3].astype(np.uint16)
        
        las.write(str(path))
    
    # ============= Hilfsmethoden =============
    
    def _detect_txt_format(self, path: Path) -> str:
        """Erkennt ob TXT-Datei CloudCompare oder XYZ Format ist"""
        with open(path, 'r') as f:
            first_line = f.readline().strip()
            
            # CloudCompare hat oft Header mit Semikolon
            if ';' in first_line:
                return '.txt'  # CloudCompare
            
            # Prüfe ob numerische Werte
            parts = first_line.split()
            if len(parts) >= 3:
                try:
                    [float(p) for p in parts[:3]]
                    return '.xyz'
                except ValueError:
                    pass
        
        return '.txt'  # Default
    
    def _is_cloudcompare_format(self, path: Path) -> bool:
        """Prüft ob Datei CloudCompare Format ist"""
        with open(path, 'r') as f:
            first_line = f.readline()
            return ';' in first_line or 'M3C2' in first_line
    
    def _resolve_path(self, path: str) -> Path:
        """Löst relativen Pfad auf"""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_path / p
    
    def convert_format(
        self,
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None
    ) -> None:
        """
        Konvertiert Punktwolke von einem Format in ein anderes.
        
        Args:
            input_path: Eingabedatei
            output_path: Ausgabedatei
            input_format: Eingabeformat (wird erkannt wenn None)
            output_format: Ausgabeformat (wird aus output_path erkannt wenn None)
        """
        # Lade Punktwolke
        points = self.load_point_cloud(input_path, format=input_format)
        
        # Speichere in neuem Format
        self.save_point_cloud(points, output_path, format=output_format)
        
        logger.info(f"Converted {input_path} to {output_path}")
    
    def ensure_format(self, base_path: str, desired_format: str = '.xyz') -> str:
        """
        Stellt sicher dass Datei im gewünschten Format vorliegt.
        Konvertiert wenn nötig.
        
        Args:
            base_path: Basis-Pfad ohne Extension
            desired_format: Gewünschtes Format
            
        Returns:
            Pfad zur Datei im gewünschten Format
        """
        base = Path(base_path)
        
        # Prüfe ob gewünschtes Format bereits existiert
        desired_path = base.with_suffix(desired_format)
        if desired_path.exists():
            return str(desired_path)
        
        # Suche nach vorhandenen Formaten
        for fmt in self.SUPPORTED_FORMATS:
            test_path = base.with_suffix(fmt)
            if test_path.exists():
                # Konvertiere
                self.convert_format(str(test_path), str(desired_path))
                return str(desired_path)
        
        raise FileNotFoundError(f"No supported format found for: {base_path}")