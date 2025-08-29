# domain/builders/pipeline_builder.py
"""Builder Pattern für Pipeline-Konfiguration"""

from typing import Optional
from domain.entities import (
    PipelineConfiguration,
    CloudPair,
    M3C2Parameters,
    OutlierConfiguration,
    ProcessingVersion
)


class PipelineBuilder:
    """Fluent Interface für Pipeline-Konfiguration"""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> 'PipelineBuilder':
        """Setzt den Builder auf Anfangszustand zurück"""
        self._cloud_pair: Optional[CloudPair] = None
        self._m3c2_params: Optional[M3C2Parameters] = None
        self._outlier_config: Optional[OutlierConfiguration] = None
        self._use_subsampled_corepoints: int = 1
        self._mov_as_corepoints: bool = True
        self._use_existing_params: bool = False
        self._only_stats: bool = False
        self._stats_type: str = "distance"
        self._project_name: str = "default_project"
        self._output_format: str = "excel"
        self._version: ProcessingVersion = ProcessingVersion.PYTHON
        return self
    
    def with_cloud_pair(self, cloud_pair: CloudPair) -> 'PipelineBuilder':
        """Setzt das CloudPair für die Pipeline"""
        self._cloud_pair = cloud_pair
        return self
    
    def with_m3c2_params(self, params: M3C2Parameters) -> 'PipelineBuilder':
        """Setzt die M3C2-Parameter"""
        self._m3c2_params = params
        return self
    
    def with_outlier_config(self, config: OutlierConfiguration) -> 'PipelineBuilder':
        """Setzt die Outlier-Konfiguration"""
        self._outlier_config = config
        return self
    
    def with_processing_options(
        self,
        use_existing_params: bool = None,
        only_stats: bool = None,
        mov_as_corepoints: bool = None,
        use_subsampled_corepoints: int = None
    ) -> 'PipelineBuilder':
        """Setzt Processing-Optionen"""
        if use_existing_params is not None:
            self._use_existing_params = use_existing_params
        if only_stats is not None:
            self._only_stats = only_stats
        if mov_as_corepoints is not None:
            self._mov_as_corepoints = mov_as_corepoints
        if use_subsampled_corepoints is not None:
            self._use_subsampled_corepoints = use_subsampled_corepoints
        return self
    
    def with_output_options(
        self,
        output_format: str = None,
        stats_type: str = None
    ) -> 'PipelineBuilder':
        """Setzt Output-Optionen"""
        if output_format is not None:
            self._output_format = output_format
        if stats_type is not None:
            self._stats_type = stats_type
        return self
    
    def with_project(self, project_name: str) -> 'PipelineBuilder':
        """Setzt den Projektnamen"""
        self._project_name = project_name
        return self
    
    def with_version(self, version: ProcessingVersion) -> 'PipelineBuilder':
        """Setzt die Processing-Version"""
        self._version = version
        return self
    
    def build(self) -> PipelineConfiguration:
        """
        Baut die finale PipelineConfiguration.
        
        Raises:
            ValueError: Wenn erforderliche Felder fehlen
        """
        if self._cloud_pair is None:
            raise ValueError("CloudPair is required")
        
        # Erstelle Default OutlierConfiguration wenn nicht gesetzt
        if self._outlier_config is None:
            self._outlier_config = OutlierConfiguration()
        
        return PipelineConfiguration(
            cloud_pair=self._cloud_pair,
            m3c2_params=self._m3c2_params,
            outlier_config=self._outlier_config,
            use_subsampled_corepoints=self._use_subsampled_corepoints,
            mov_as_corepoints=self._mov_as_corepoints,
            use_existing_params=self._use_existing_params,
            only_stats=self._only_stats,
            stats_type=self._stats_type,
            project_name=self._project_name,
            output_format=self._output_format,
            version=self._version
        )