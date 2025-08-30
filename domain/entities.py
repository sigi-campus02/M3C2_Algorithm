# domain/entities.py
"""Domain Entities und Value Objects"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import numpy as np
from pathlib import Path


class ProcessingVersion(Enum):
    """Verarbeitungsversionen"""
    PYTHON = "python"
    CLOUDCOMPARE = "CC"


class OutlierMethod(Enum):
    """Methoden zur Ausreißer-Erkennung"""
    RMSE = "rmse"
    MAD = "mad"
    IQR = "iqr"
    ZSCORE = "zscore"


class CloudType(Enum):
    """Typ der Punktwolke"""
    PLAIN = "plain"          # Original ohne AI
    AI_PROCESSED = "ai"      # Mit AI verarbeitet


class ComparisonCase(Enum):
    """Vergleichsfälle für Punktwolken"""
    CASE1 = "plain_vs_plain"     # a-i vs b-i
    CASE2 = "plain_vs_ai"        # a-i vs b-i-AI
    CASE3 = "ai_vs_plain"        # a-i-AI vs b-i
    CASE4 = "ai_vs_ai"           # a-i-AI vs b-i-AI


@dataclass(frozen=True)
class M3C2Parameters:
    """M3C2 Algorithmus-Parameter"""
    normal_scale: float
    search_scale: float
    
    def validate(self) -> bool:
        """Validiert die Parameter"""
        return (
            self.normal_scale > 0 and 
            self.search_scale > 0 and
            self.search_scale >= self.normal_scale
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Konvertiert zu Dictionary für Persistierung"""
        return {
            "NormalScale": self.normal_scale,
            "SearchScale": self.search_scale
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'M3C2Parameters':
        """Erstellt aus Dictionary"""
        return cls(
            normal_scale=data.get("NormalScale", 0.002),
            search_scale=data.get("SearchScale", 0.004)
        )


@dataclass(frozen=True)
class CloudPair:
    """Punktwolken-Paar für Vergleich"""
    moving_cloud: str
    reference_cloud: str
    folder_id: str
    comparison_case: ComparisonCase
    index: int  # Der numerische Index (1, 2, 3, etc.)
    
    @property
    def tag(self) -> str:
        """Eindeutiger Tag für dieses Paar"""
        return f"{self.moving_cloud}-{self.reference_cloud}"
    
    def get_output_prefix(self, version: ProcessingVersion) -> str:
        """Gibt Präfix für Output-Dateien zurück"""
        return f"{version.value}_{self.tag}"


@dataclass
class OutlierConfiguration:
    """Konfiguration für Ausreißer-Erkennung"""
    method: OutlierMethod = OutlierMethod.RMSE
    multiplier: float = 3.0
    
    def validate(self) -> bool:
        """Validiert die Konfiguration"""
        return self.multiplier > 0


@dataclass
class PipelineConfiguration:
    """Vollständige Pipeline-Konfiguration"""
    cloud_pair: CloudPair
    m3c2_params: Optional[M3C2Parameters] = None
    outlier_config: OutlierConfiguration = field(default_factory=OutlierConfiguration)
    use_subsampled_corepoints: int = 1
    mov_as_corepoints: bool = True
    use_existing_params: bool = False
    only_stats: bool = False
    stats_type: str = "distance"  # "distance" oder "single"
    project_name: str = "default_project"
    output_format: str = "excel"
    version: ProcessingVersion = ProcessingVersion.PYTHON

    # Zusätzliche Felder für erweiterte Konfiguration
    folder_id: Optional[str] = None
    generate_plots: bool = False
    output_base_path: Optional[Any] = None  # Path oder str

    def __post_init__(self):
        """Post-Init für automatische Felder"""
        # Setze folder_id aus cloud_pair wenn nicht explizit gesetzt
        if self.folder_id is None and hasattr(self.cloud_pair, 'folder_id'):
            self.folder_id = self.cloud_pair.folder_id

        # Setze default output_base_path wenn nicht gesetzt
        if self.output_base_path is None:
            self.output_base_path = f"outputs/{self.project_name}_output"

        # Konvertiere zu Path wenn es ein String ist
        if isinstance(self.output_base_path, str):
            from pathlib import Path
            self.output_base_path = Path(self.output_base_path)

    def validate(self) -> bool:
        """Validiert die gesamte Konfiguration"""
        validations = [
            self.use_subsampled_corepoints > 0,
            self.outlier_config.validate(),
            self.output_format in ["excel", "json", "csv", "all"],
            self.stats_type in ["distance", "single"]
        ]

        if self.m3c2_params:
            validations.append(self.m3c2_params.validate())

        return all(validations)

    @property
    def plots_path(self) -> str:
        """Pfad für Plots"""
        return str(self.output_base_path / f"{self.project_name}_plots")


@dataclass
class M3C2Result:
    """Ergebnis einer M3C2-Berechnung"""
    distances: np.ndarray
    uncertainties: np.ndarray
    parameters_used: M3C2Parameters
    nan_percentage: float
    valid_count: int
    
    @classmethod
    def from_arrays(
        cls, 
        distances: np.ndarray,
        uncertainties: np.ndarray,
        parameters: M3C2Parameters
    ) -> 'M3C2Result':
        """Erstellt aus Numpy-Arrays"""
        nan_count = np.isnan(distances).sum()
        total_count = len(distances)
        
        return cls(
            distances=distances,
            uncertainties=uncertainties,
            parameters_used=parameters,
            nan_percentage=(nan_count / total_count * 100) if total_count > 0 else 0,
            valid_count=total_count - nan_count
        )


@dataclass
class CloudStatistics:
    """Statistiken für eine Punktwolke"""
    folder_id: str
    version: str
    timestamp: str
    
    # Basis-Metriken
    total_points: int
    nan_count: int
    valid_count: int
    
    # Statistische Kennzahlen
    mean: float
    median: float
    std: float
    rms: float
    mae: float
    nmad: float
    
    # Quantile
    q05: float
    q25: float
    q75: float
    q95: float
    iqr: float
    
    # Outlier-Informationen
    outlier_count: int = 0
    inlier_count: int = 0
    outlier_method: Optional[str] = None
    outlier_threshold: Optional[float] = None
    
    # Erweiterte Statistiken
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für Export"""
        return {
            "Folder": self.folder_id,
            "Version": self.version,
            "Timestamp": self.timestamp,
            "Total Points": self.total_points,
            "NaN Count": self.nan_count,
            "Valid Count": self.valid_count,
            "Mean": self.mean,
            "Median": self.median,
            "Std": self.std,
            "RMS": self.rms,
            "MAE": self.mae,
            "NMAD": self.nmad,
            "Q05": self.q05,
            "Q25": self.q25,
            "Q75": self.q75,
            "Q95": self.q95,
            "IQR": self.iqr,
            "Outlier Count": self.outlier_count,
            "Inlier Count": self.inlier_count,
            "Outlier Method": self.outlier_method,
            "Outlier Threshold": self.outlier_threshold,
            "Skewness": self.skewness,
            "Kurtosis": self.kurtosis
        }