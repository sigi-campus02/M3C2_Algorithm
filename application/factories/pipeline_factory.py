# application/factories/pipeline_factory.py
"""Factory für Pipeline-Erstellung"""

import logging
from typing import List

from domain.commands.base import Command, CompositeCommand, ConditionalCommand
from domain.commands.m3c2_commands import (
    LoadPointCloudsCommand,
    EstimateParametersCommand,
    RunM3C2Command,
    DetectOutliersCommand,
    SaveResultsCommand,
    ComputeStatisticsCommand,
    GenerateVisualizationCommand
)
from domain.entities import PipelineConfiguration
from application.factories.service_factory import ServiceFactory

# Import Pipeline aus dem gleichen Package
from application.orchestration.pipeline_orchestrator import Pipeline

logger = logging.getLogger(__name__)


class PipelineFactory:
    """Factory zur Erstellung von Pipelines basierend auf Konfiguration"""

    def __init__(self, service_factory: ServiceFactory):
        self.service_factory = service_factory
        logger.debug("PipelineFactory initialized")

    def create_pipeline(self, config: PipelineConfiguration) -> Pipeline:
        """
        Erstellt eine Pipeline basierend auf der Konfiguration.

        Args:
            config: Pipeline-Konfiguration

        Returns:
            Konfigurierte Pipeline
        """
        if config.only_stats:
            # Nur Statistiken berechnen
            commands = self._create_statistics_only_commands(config)
        else:
            # Volle Pipeline
            commands = self._create_full_pipeline_commands(config)

        pipeline_name = f"Pipeline_{config.cloud_pair.tag}"
        return Pipeline(commands, name=pipeline_name)

    def _create_full_pipeline_commands(self, config: PipelineConfiguration) -> List[Command]:
        """Erstellt Commands für die volle Pipeline"""
        commands = []

        # 1. Lade Punktwolken
        pc_repo = self.service_factory.get_point_cloud_repository()
        commands.append(LoadPointCloudsCommand(pc_repo))

        # 2. Schätze Parameter (wenn nicht vorhanden)
        if not config.m3c2_params:
            param_estimator = self.service_factory.get_param_estimator()
            # Konfiguriere Strategie basierend auf Config
            sample_size = self.service_factory.get_config_value(
                'processing.sample_size', 10000
            )
            # Hier müsste RadiusScanStrategy importiert werden
            from domain.strategies.parameter_estimation import RadiusScanStrategy
            strategy = RadiusScanStrategy(sample_size=sample_size)

            commands.append(EstimateParametersCommand(param_estimator, strategy))

        # 3. Führe M3C2 aus
        m3c2_runner = self.service_factory.get_m3c2_runner()
        commands.append(RunM3C2Command(m3c2_runner))

        # 4. Erkenne Ausreißer
        outlier_strategy = self.service_factory.get_outlier_strategy(
            config.outlier_config.method,
            config.outlier_config.multiplier
        )
        commands.append(DetectOutliersCommand(outlier_strategy))

        # 5. Berechne Statistiken
        stats_service = self.service_factory.get_statistics_service()
        commands.append(ComputeStatisticsCommand(stats_service))

        # 6. Generiere Visualisierungen (optional)
        if config.generate_plots:
            vis_service = self.service_factory.get_visualization_service()
            commands.append(GenerateVisualizationCommand(vis_service))

        # 7. Speichere Ergebnisse
        result_repo = self.service_factory.get_result_repository()
        commands.append(SaveResultsCommand(result_repo))

        return commands

    def _create_statistics_only_commands(self, config: PipelineConfiguration) -> List[Command]:
        """Erstellt Commands für reine Statistik-Berechnung"""
        commands = []

        # 1. Lade vorhandene Ergebnisse
        result_repo = self.service_factory.get_result_repository()
        # Hier würde ein LoadExistingResultsCommand verwendet
        # commands.append(LoadExistingResultsCommand(result_repo))

        # 2. Berechne Statistiken
        stats_service = self.service_factory.get_statistics_service()
        commands.append(ComputeStatisticsCommand(stats_service))

        # 3. Speichere aktualisierte Statistiken
        commands.append(SaveResultsCommand(result_repo))

        return commands