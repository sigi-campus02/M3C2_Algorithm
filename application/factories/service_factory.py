# application/factories/service_factory.py
"""Service Factory mit Dependency Injection und Lazy Loading"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from domain.entities import OutlierMethod
from domain.strategies.outlier_detection import (
    OutlierDetectionStrategy,
    RMSEOutlierStrategy,
    MADOutlierStrategy,
    IQROutlierStrategy,
    ZScoreOutlierStrategy
)
from infrastructure.repositories.point_cloud_repository import PointCloudRepository
from infrastructure.repositories.distance_repository import DistanceRepository
from infrastructure.repositories.file_point_cloud_repository import (
    FileParameterRepository,
    FileStatisticsRepository
)

logger = logging.getLogger(__name__)


class ServiceFactory:
    """
    Factory für Services mit Lazy Loading und Singleton Pattern.

    Diese Factory verwaltet alle Services und Repositories der Anwendung
    und stellt sicher, dass jeder Service nur einmal instantiiert wird.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert die ServiceFactory mit Konfiguration.

        Args:
            config: Dictionary mit Konfigurationsparametern
        """
        self.config = config
        self._repositories: Dict[str, Any] = {}
        self._services: Dict[str, Any] = {}
        self._strategies: Dict[str, Any] = {}

        # Extrahiere wichtige Pfade aus Config
        self.data_path = Path(config.get('data_path', 'data'))
        self.output_path = Path(config.get('output_path', 'outputs'))

        logger.info(f"ServiceFactory initialized with data_path={self.data_path}, output_path={self.output_path}")

    # ============= Repositories =============

    def get_point_cloud_repository(self) -> PointCloudRepository:
        """Gibt PointCloudRepository zurück (Singleton)"""
        if 'point_cloud' not in self._repositories:
            self._repositories['point_cloud'] = PointCloudRepository(str(self.data_path))
            logger.debug("Created PointCloudRepository")
        return self._repositories['point_cloud']

    def get_distance_repository(self) -> DistanceRepository:
        """Gibt DistanceRepository zurück (Singleton)"""
        if 'distance' not in self._repositories:
            self._repositories['distance'] = DistanceRepository(str(self.output_path))
            logger.debug("Created DistanceRepository")
        return self._repositories['distance']

    def get_parameter_repository(self) -> FileParameterRepository:
        """Gibt FileParameterRepository zurück (Singleton)"""
        if 'parameter' not in self._repositories:
            self._repositories['parameter'] = FileParameterRepository(str(self.data_path))
            logger.debug("Created FileParameterRepository")
        return self._repositories['parameter']

    def get_statistics_repository(self) -> FileStatisticsRepository:
        """Gibt FileStatisticsRepository zurück (Singleton)"""
        if 'statistics' not in self._repositories:
            self._repositories['statistics'] = FileStatisticsRepository(str(self.output_path))
            logger.debug("Created FileStatisticsRepository")
        return self._repositories['statistics']

    def get_result_repository(self):
        """Alias für distance repository"""
        return self.get_distance_repository()

    # ============= Strategies =============

    def get_outlier_strategy(
            self,
            method: OutlierMethod,
            multiplier: float = 3.0
    ) -> OutlierDetectionStrategy:
        """Gibt die passende Outlier-Detection Strategie zurück"""

        key = f"{method.value}_{multiplier}"

        if key not in self._strategies:
            if method == OutlierMethod.RMSE:
                strategy = RMSEOutlierStrategy(multiplier)
            elif method == OutlierMethod.MAD:
                strategy = MADOutlierStrategy(multiplier)
            elif method == OutlierMethod.IQR:
                strategy = IQROutlierStrategy(multiplier)
            elif method == OutlierMethod.ZSCORE:
                strategy = ZScoreOutlierStrategy(multiplier)
            else:
                raise ValueError(f"Unknown outlier method: {method}")

            self._strategies[key] = strategy
            logger.debug(f"Created {method.value} outlier strategy with multiplier={multiplier}")

        return self._strategies[key]

    # ============= Services =============

    def get_m3c2_runner(self):
        """Gibt M3C2Runner zurück"""
        if 'm3c2_runner' not in self._services:
            # KORRIGIERT: Richtiger Import-Pfad
            from application.orchestration.m3c2_runner import M3C2Runner
            self._services['m3c2_runner'] = M3C2Runner()
            logger.debug("Created M3C2Runner")
        return self._services['m3c2_runner']

    def get_param_estimator(self):
        """Gibt ParamEstimator zurück"""
        if 'param_estimator' not in self._services:
            # KORRIGIERT: Richtiger Import-Pfad
            from application.services.param_estimator import ParamEstimator
            self._services['param_estimator'] = ParamEstimator()
            logger.debug("Created ParamEstimator")
        return self._services['param_estimator']

    def get_statistics_service(self):
        """Gibt StatisticsService zurück"""
        if 'statistics_service' not in self._services:
            # KORRIGIERT: Richtiger Import-Pfad
            from application.services.statistics_service import StatisticsService
            self._services['statistics_service'] = StatisticsService()
            logger.debug("Created StatisticsService")
        return self._services['statistics_service']

    def get_visualization_service(self):
        """Gibt VisualizationService zurück"""
        if 'visualization_service' not in self._services:
            # KORRIGIERT: Richtiger Import-Pfad
            from application.services.plot_service import PlotService
            self._services['visualization_service'] = PlotService()
            logger.debug("Created PlotService as VisualizationService")
        return self._services['visualization_service']

    def get_plot_service(self):
        """Gibt PlotService zurück"""
        if 'plot_service' not in self._services:
            # KORRIGIERT: Richtiger Import-Pfad
            from application.services.plot_service import PlotService
            plot_config = self.config.get('plotting', {})
            self._services['plot_service'] = PlotService()
            logger.debug("Created PlotService")
        return self._services['plot_service']

    def get_export_service(self):
        """Gibt ExportService zurück"""
        if 'export_service' not in self._services:
            # KORRIGIERT: Richtiger Import-Pfad
            from application.services.export_service import ExportService
            self._services['export_service'] = ExportService()
            logger.debug("Created ExportService")
        return self._services['export_service']

    def get_report_service(self):
        """Gibt ReportService zurück"""
        if 'report_service' not in self._services:
            # KORRIGIERT: Richtiger Import-Pfad
            from application.services.report_service import ReportService
            self._services['report_service'] = ReportService()
            logger.debug("Created ReportService")
        return self._services['report_service']

    def get_cloud_pair_scanner(self):
        """Gibt CloudPairScanner Service zurück (Singleton)."""
        if 'cloud_pair_scanner' not in self._services:
            from application.services.cloud_pair_scanner import CloudPairScanner, ScannerConfig

            # Erstelle Scanner-Konfiguration
            scanner_config = ScannerConfig()

            # Überschreibe mit custom config wenn vorhanden
            if 'scanner' in self.config:
                scanner_settings = self.config['scanner']
                if 'supported_formats' in scanner_settings:
                    scanner_config.supported_formats = scanner_settings['supported_formats']
                if 'naming_conventions' in scanner_settings:
                    scanner_config.naming_conventions.update(scanner_settings['naming_conventions'])

            self._services['cloud_pair_scanner'] = CloudPairScanner(scanner_config)
            logger.debug("Created CloudPairScanner service")

        return self._services['cloud_pair_scanner']

    def get_visualization_service(self):
        """Gibt VisualizationService zurück"""
        if 'visualization_service' not in self._services:
            try:
                # Versuche erweiterten Service zu laden
                from application.services.visualization_service import VisualizationService
                self._services['visualization_service'] = VisualizationService(
                    repository=self.get_point_cloud_repository()
                )
                logger.debug("Created VisualizationService")
            except ImportError:
                # Fallback auf normalen Visualization Service
                logger.warning("VisualizationService not found, using standard VisualizationService")
                return self.get_visualization_service()
        return self._services['visualization_service']

    # ============= Configuration =============

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Holt einen Konfigurationswert"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    # ============= Cleanup =============

    def reset(self):
        """Setzt alle Services und Repositories zurück"""
        self._repositories.clear()
        self._services.clear()
        self._strategies.clear()
        logger.info("Reset all services and repositories")

    def __del__(self):
        """Cleanup bei Zerstörung"""
        self.reset()