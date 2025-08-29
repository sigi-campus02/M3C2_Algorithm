# domain/validators/point_cloud_validators.py
"""Validator-Implementierungen für Punktwolken und Pipeline"""

import logging
from typing import Optional, List, Any
import numpy as np
from pathlib import Path

from domain.validators.base import Validator

logger = logging.getLogger(__name__)


class PointCountValidator(Validator):
    """Validiert die Anzahl der Punkte in einer Punktwolke"""
    
    def __init__(self, min_points: int = 1000, max_points: Optional[int] = None):
        super().__init__()
        self.min_points = min_points
        self.max_points = max_points
    
    def _validate_impl(self, cloud: np.ndarray) -> bool:
        """Prüft ob Punktanzahl im gültigen Bereich liegt"""
        point_count = len(cloud)
        
        if point_count < self.min_points:
            logger.error(f"Point cloud has too few points: {point_count} < {self.min_points}")
            return False
        
        if self.max_points and point_count > self.max_points:
            logger.error(f"Point cloud has too many points: {point_count} > {self.max_points}")
            return False
        
        logger.debug(f"Point count valid: {point_count}")
        return True


class DimensionValidator(Validator):
    """Validiert die Dimensionen einer Punktwolke"""
    
    def __init__(self, expected_dims: int = 3):
        super().__init__()
        self.expected_dims = expected_dims
    
    def _validate_impl(self, cloud: np.ndarray) -> bool:
        """Prüft ob Punktwolke die erwarteten Dimensionen hat"""
        if len(cloud.shape) != 2:
            logger.error(f"Point cloud has wrong shape: {cloud.shape}")
            return False
        
        dims = cloud.shape[1]
        if dims < self.expected_dims:
            logger.error(f"Point cloud has too few dimensions: {dims} < {self.expected_dims}")
            return False
        
        logger.debug(f"Dimensions valid: {dims}")
        return True


class BoundingBoxValidator(Validator):
    """Validiert die Bounding Box einer Punktwolke"""
    
    def __init__(self, max_extent: float = 1000.0):
        super().__init__()
        self.max_extent = max_extent
    
    def _validate_impl(self, cloud: np.ndarray) -> bool:
        """Prüft ob Punktwolke in vernünftiger Bounding Box liegt"""
        if len(cloud) == 0:
            return True
        
        # Berechne Bounding Box
        min_coords = np.min(cloud[:, :3], axis=0)
        max_coords = np.max(cloud[:, :3], axis=0)
        extent = max_coords - min_coords
        
        max_dim = np.max(extent)
        
        if max_dim > self.max_extent:
            logger.warning(f"Point cloud extent exceeds limit: {max_dim:.2f} > {self.max_extent}")
            return False
        
        logger.debug(f"Bounding box valid: extent={extent}")
        return True


class DensityValidator(Validator):
    """Validiert die Punktdichte einer Punktwolke"""
    
    def __init__(self, min_density: float = 0.1, sample_size: int = 1000):
        super().__init__()
        self.min_density = min_density
        self.sample_size = sample_size
    
    def _validate_impl(self, cloud: np.ndarray) -> bool:
        """Prüft ob Punktdichte ausreichend ist"""
        if len(cloud) < self.sample_size:
            sample = cloud
        else:
            # Zufälliges Sample für Effizienz
            indices = np.random.choice(len(cloud), self.sample_size, replace=False)
            sample = cloud[indices]
        
        # Berechne durchschnittlichen Punktabstand
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=2).fit(sample[:, :3])
        distances, _ = nbrs.kneighbors(sample[:, :3])
        avg_spacing = np.mean(distances[:, 1])
        
        # Density ist reziprok zum Spacing
        density = 1.0 / avg_spacing if avg_spacing > 0 else 0
        
        if density < self.min_density:
            logger.warning(f"Point cloud density too low: {density:.3f} < {self.min_density}")
            return False
        
        logger.debug(f"Density valid: {density:.3f} points/unit")
        return True


class FileExistenceValidator(Validator):
    """Validiert die Existenz von Dateien"""
    
    def __init__(self, base_path: Optional[str] = None):
        super().__init__()
        self.base_path = Path(base_path) if base_path else Path.cwd()
    
    def _validate_impl(self, file_path: str) -> bool:
        """Prüft ob Datei existiert"""
        full_path = self.base_path / file_path
        
        if not full_path.exists():
            logger.error(f"File not found: {full_path}")
            return False
        
        if not full_path.is_file():
            logger.error(f"Path is not a file: {full_path}")
            return False
        
        logger.debug(f"File exists: {full_path}")
        return True


class ParameterValidator(Validator):
    """Validiert M3C2-Parameter"""
    
    def __init__(self, min_scale: float = 0.0001, max_scale: float = 100.0):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
    
    def _validate_impl(self, params: Any) -> bool:
        """Prüft ob M3C2-Parameter gültig sind"""
        from domain.entities import M3C2Parameters
        
        if not isinstance(params, M3C2Parameters):
            logger.error(f"Invalid parameter type: {type(params)}")
            return False
        
        # Prüfe Normal Scale
        if params.normal_scale < self.min_scale or params.normal_scale > self.max_scale:
            logger.error(f"Normal scale out of range: {params.normal_scale}")
            return False
        
        # Prüfe Search Scale
        if params.search_scale < self.min_scale or params.search_scale > self.max_scale:
            logger.error(f"Search scale out of range: {params.search_scale}")
            return False
        
        # Search Scale sollte >= Normal Scale sein
        if params.search_scale < params.normal_scale:
            logger.warning(f"Search scale < normal scale: {params.search_scale} < {params.normal_scale}")
            return False
        
        logger.debug(f"Parameters valid: normal={params.normal_scale}, search={params.search_scale}")
        return True


class ConfigurationValidator(Validator):
    """Validiert eine vollständige Pipeline-Konfiguration"""
    
    def _validate_impl(self, config: Any) -> bool:
        """Prüft ob Pipeline-Konfiguration gültig ist"""
        from domain.entities import PipelineConfiguration
        
        if not isinstance(config, PipelineConfiguration):
            logger.error(f"Invalid configuration type: {type(config)}")
            return False
        
        # Nutze die eingebaute Validierung
        if not config.validate():
            logger.error("Configuration validation failed")
            return False
        
        logger.debug("Configuration valid")
        return True


class CloudPairValidator(Validator):
    """Validiert ein Punktwolken-Paar"""
    
    def __init__(self, check_overlap: bool = True):
        super().__init__()
        self.check_overlap = check_overlap
    
    def _validate_impl(self, pair: tuple) -> bool:
        """Prüft ob Punktwolken-Paar kompatibel ist"""
        if len(pair) != 2:
            logger.error("Cloud pair must contain exactly 2 clouds")
            return False
        
        cloud1, cloud2 = pair
        
        # Prüfe Dimensionen
        if cloud1.shape[1] != cloud2.shape[1]:
            logger.error(f"Dimension mismatch: {cloud1.shape[1]} != {cloud2.shape[1]}")
            return False
        
        if self.check_overlap:
            # Prüfe grobe Überlappung anhand Bounding Boxes
            bb1_min = np.min(cloud1[:, :3], axis=0)
            bb1_max = np.max(cloud1[:, :3], axis=0)
            bb2_min = np.min(cloud2[:, :3], axis=0)
            bb2_max = np.max(cloud2[:, :3], axis=0)
            
            # Prüfe ob Boxes überlappen
            overlap = np.all(bb1_max >= bb2_min) and np.all(bb2_max >= bb1_min)
            
            if not overlap:
                logger.warning("Point clouds have no bounding box overlap")
                return False
        
        logger.debug("Cloud pair valid")
        return True


def create_validation_chain() -> Validator:
    """
    Erstellt eine Standard-Validierungskette für Punktwolken.
    
    Returns:
        Verkettete Validatoren
    """
    # Erstelle Validator-Kette
    validator = PointCountValidator(min_points=100)
    validator.set_next(DimensionValidator(expected_dims=3))\
             .set_next(BoundingBoxValidator(max_extent=1000.0))\
             .set_next(DensityValidator(min_density=0.01))
    
    return validator


def validate_pipeline_input(
    mov_cloud: np.ndarray,
    ref_cloud: np.ndarray,
    params: Optional[Any] = None
) -> List[str]:
    """
    Validiert die Eingaben für die Pipeline.
    
    Args:
        mov_cloud: Moving Punktwolke
        ref_cloud: Reference Punktwolke
        params: Optionale M3C2-Parameter
        
    Returns:
        Liste von Fehlermeldungen (leer wenn alles ok)
    """
    errors = []
    
    # Validiere einzelne Clouds
    validator = create_validation_chain()
    
    if not validator.validate(mov_cloud):
        errors.append("Moving cloud validation failed")
    
    if not validator.validate(ref_cloud):
        errors.append("Reference cloud validation failed")
    
    # Validiere Cloud-Paar
    pair_validator = CloudPairValidator()
    if not pair_validator.validate((mov_cloud, ref_cloud)):
        errors.append("Cloud pair validation failed")
    
    # Validiere Parameter wenn vorhanden
    if params:
        param_validator = ParameterValidator()
        if not param_validator.validate(params):
            errors.append("Parameter validation failed")
    
    return errors