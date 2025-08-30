# domain/builders/pipeline_builder.py
"""Builder Pattern für Pipeline-Konfiguration"""

from typing import Optional, Dict, Any
from domain.entities import (
    PipelineConfiguration,
    CloudPair,
    M3C2Parameters,
    OutlierConfiguration,
    OutlierMethod,
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
        self._folder_id: Optional[str] = None
        self._additional_options: Dict[str, Any] = {}  # Für zusätzliche Optionen
        return self

    def with_cloud_pair(self, cloud_pair: CloudPair) -> 'PipelineBuilder':
        """Setzt das CloudPair für die Pipeline"""
        self._cloud_pair = cloud_pair
        # Automatisch folder_id aus CloudPair übernehmen wenn vorhanden
        if hasattr(cloud_pair, 'folder_id') and cloud_pair.folder_id:
            self._folder_id = cloud_pair.folder_id
        return self

    def with_m3c2_params(self, params: M3C2Parameters) -> 'PipelineBuilder':
        """Setzt die M3C2-Parameter"""
        self._m3c2_params = params
        return self

    def with_outlier_config(
            self,
            method: OutlierMethod,
            multiplier: float = 3.0
    ) -> 'PipelineBuilder':
        """
        Setzt die Outlier-Konfiguration.

        Args:
            method: OutlierMethod Enum oder String
            multiplier: Multiplikator für Outlier-Detection
        """
        # Konvertiere String zu Enum falls nötig
        if isinstance(method, str):
            method = OutlierMethod(method)

        self._outlier_config = OutlierConfiguration(
            method=method,
            multiplier=multiplier
        )
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

    def with_output_format(self, output_format: str) -> 'PipelineBuilder':
        """Convenience-Methode für Output-Format"""
        self._output_format = output_format
        return self

    def with_project(self, project_name: str) -> 'PipelineBuilder':
        """Setzt den Projektnamen"""
        self._project_name = project_name
        return self

    def with_project_name(self, project_name: str) -> 'PipelineBuilder':
        """Alias für with_project für bessere Lesbarkeit"""
        return self.with_project(project_name)

    def with_folder_id(self, folder_id: str) -> 'PipelineBuilder':
        """Setzt die Folder ID"""
        self._folder_id = folder_id
        return self

    def with_version(self, version: ProcessingVersion) -> 'PipelineBuilder':
        """Setzt die Processing-Version"""
        self._version = version
        return self

    def with_options(self, options: Dict[str, Any]) -> 'PipelineBuilder':
        """
        Setzt zusätzliche Optionen als Dictionary.

        Args:
            options: Dictionary mit zusätzlichen Optionen
        """
        if options:
            # Verarbeite bekannte Optionen
            if 'only_stats' in options:
                self._only_stats = options['only_stats']
            if 'use_existing_params' in options:
                self._use_existing_params = options['use_existing_params']
            if 'output_format' in options:
                self._output_format = options['output_format']
            if 'stats_type' in options:
                self._stats_type = options['stats_type']

            # Speichere alle anderen Optionen
            self._additional_options.update(options)

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

        # Erstelle PipelineConfiguration
        config = PipelineConfiguration(
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

        # Füge zusätzliche Optionen als Attribute hinzu, wenn sie existieren
        for key, value in self._additional_options.items():
            if not hasattr(config, key):
                setattr(config, key, value)

        return config

    def validate(self) -> bool:
        """
        Validiert die aktuelle Konfiguration.

        Returns:
            True wenn gültig, sonst False
        """
        if self._cloud_pair is None:
            return False

        if self._outlier_config and not self._outlier_config.validate():
            return False

        if self._m3c2_params and not self._m3c2_params.validate():
            return False

        return True