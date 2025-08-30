# application/factories/service_factory.py
"""Service Factory für Dependency Injection"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from application.services.cloud_pair_scanner import CloudPairScanner
from infrastructure.repositories.file_point_cloud_repository import (
    FilePointCloudRepository,
    FileParameterRepository,
    FileStatisticsRepository
)
from domain.strategies.outlier_detection import (
    OutlierDetectionStrategy,
    RMSEOutlierStrategy,
    MADOutlierStrategy,
    IQROutlierStrategy,
    ZScoreOutlierStrategy
)
from domain.entities import OutlierMethod

logger = logging.getLogger(__name__)


class ServiceFactory:
    """Factory für Service-Erstellung mit Dependency Injection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._repositories = {}
        self._services = {}
        self._strategies = {}
        
        # Initialisiere Basis-Pfade
        self.data_path = Path(config.get('data_path', 'data'))
        self.output_path = Path(config.get('output_path', 'outputs'))
        
        logger.info(f"ServiceFactory initialized with data_path={self.data_path}, output_path={self.output_path}")
    
    # ============= Repositories =============
    
    def get_point_cloud_repository(self) -> FilePointCloudRepository:
        """Gibt PointCloud Repository zurück (Singleton)"""
        if 'point_cloud_repo' not in self._repositories:
            self._repositories['point_cloud_repo'] = FilePointCloudRepository(
                base_path=str(self.data_path)
            )
            logger.debug("Created FilePointCloudRepository")
        return self._repositories['point_cloud_repo']
    
    def get_parameter_repository(self) -> FileParameterRepository:
        """Gibt Parameter Repository zurück (Singleton)"""
        if 'param_repo' not in self._repositories:
            self._repositories['param_repo'] = FileParameterRepository(
                base_path=str(self.data_path)
            )
            logger.debug("Created FileParameterRepository")
        return self._repositories['param_repo']
    
    def get_statistics_repository(self) -> FileStatisticsRepository:
        """Gibt Statistics Repository zurück (Singleton)"""
        if 'stats_repo' not in self._repositories:
            self._repositories['stats_repo'] = FileStatisticsRepository(
                output_path=str(self.output_path)
            )
            logger.debug("Created FileStatisticsRepository")
        return self._repositories['stats_repo']
    
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
            from orchestration.m3c2_runner import M3C2Runner
            self._services['m3c2_runner'] = M3C2Runner()
            logger.debug("Created M3C2Runner")
        return self._services['m3c2_runner']
    
    def get_param_estimator(self):
        """Gibt ParamEstimator zurück"""
        if 'param_estimator' not in self._services:
            from services.param_estimator import ParamEstimator
            self._services['param_estimator'] = ParamEstimator()
            logger.debug("Created ParamEstimator")
        return self._services['param_estimator']
    
    def get_statistics_service(self):
        """Gibt StatisticsService zurück"""
        if 'statistics_service' not in self._services:
            from application.services.statistics_service import StatisticsService
            self._services['statistics_service'] = StatisticsService(
                repository=self.get_statistics_repository()
            )
            logger.debug("Created StatisticsService")
        return self._services['statistics_service']
    
    def get_visualization_service(self):
        """Gibt VisualizationService zurück"""
        if 'visualization_service' not in self._services:
            from application.services.visualization_service import VisualizationService
            self._services['visualization_service'] = VisualizationService(
                repository=self.get_point_cloud_repository()
            )
            logger.debug("Created VisualizationService")
        return self._services['visualization_service']
    
    def get_plot_service(self):
        """Gibt PlotService zurück"""
        if 'plot_service' not in self._services:
            from application.services.plot_service import PlotService
            plot_config = self.config.get('plotting', {})
            self._services['plot_service'] = PlotService(
                repository=self.get_point_cloud_repository(),
                config=plot_config
            )
            logger.debug("Created PlotService")
        return self._services['plot_service']
    
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
    
    def update_config(self, key: str, value: Any) -> None:
        """Aktualisiert einen Konfigurationswert"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"Updated config: {key} = {value}")
    
    def reset_services(self) -> None:
        """Setzt alle Services zurück (für Tests)"""
        self._repositories.clear()
        self._services.clear()
        self._strategies.clear()
        logger.info("Reset all services and repositories")

    def get_cloud_pair_scanner(self):
        """
        Gibt CloudPairScanner Service zurück (Singleton).

        Returns:
            CloudPairScanner Instanz
        """
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