# application/factories/pipeline_factory.py
"""Factory für Pipeline-Erstellung"""

import logging
from typing import List

from domain.commands.base import Command, CompositeCommand, ConditionalCommand
from domain.commands.m3c2_commands import (
    LoadPointCloudsCommand,  # Ohne V2!
    EstimateParametersCommand,
    RunM3C2Command,
    DetectOutliersCommand,
    SaveResultsCommand,
    ComputeStatisticsCommand,
    GenerateVisualizationCommand
)
from domain.entities import PipelineConfiguration
from application.orchestration.pipeline_orchestrator import Pipeline
from application.factories.service_factory import ServiceFactory

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
        
        # 5. Speichere Ergebnisse
        param_repo = self.service_factory.get_parameter_repository()
        commands.append(SaveResultsCommand(pc_repo, param_repo))
        
        # 6. Berechne Statistiken
        stats_service = self.service_factory.get_statistics_service()
        commands.append(ComputeStatisticsCommand(stats_service))
        
        # 7. Generiere Visualisierungen (optional)
        if self._should_create_visualizations(config):
            viz_service = self.service_factory.get_visualization_service()
            commands.append(GenerateVisualizationCommand(viz_service))
        
        logger.debug(f"Created full pipeline with {len(commands)} commands")
        return commands
    
    def _create_statistics_only_commands(self, config: PipelineConfiguration) -> List[Command]:
        """Erstellt Commands für reine Statistik-Berechnung"""
        commands = []
        
        # Lade existierende Ergebnisse
        commands.append(LoadExistingResultsCommand(
            self.service_factory.get_point_cloud_repository()
        ))
        
        # Berechne Statistiken
        stats_service = self.service_factory.get_statistics_service()
        commands.append(ComputeStatisticsCommand(stats_service))
        
        logger.debug(f"Created statistics-only pipeline with {len(commands)} commands")
        return commands
    
    def _should_create_visualizations(self, config: PipelineConfiguration) -> bool:
        """Bestimmt ob Visualisierungen erstellt werden sollen"""
        # Kann aus Config gelesen werden
        return self.service_factory.get_config_value(
            'visualization.enabled', True
        )
    
    def create_conditional_pipeline(
        self,
        config: PipelineConfiguration,
        conditions: dict
    ) -> Pipeline:
        """
        Erstellt eine Pipeline mit bedingten Commands.
        
        Args:
            config: Pipeline-Konfiguration
            conditions: Dictionary mit Bedingungen für Commands
            
        Returns:
            Pipeline mit bedingten Commands
        """
        commands = []
        
        # Basis-Commands (immer ausführen)
        pc_repo = self.service_factory.get_point_cloud_repository()
        commands.append(LoadPointCloudsCommand(pc_repo))
        
        # Bedingte Parameter-Schätzung
        if conditions.get('estimate_params', True):
            param_cmd = self._create_parameter_estimation_command()
            condition_func = lambda ctx: not ctx.has('m3c2_params')
            commands.append(ConditionalCommand(
                "ConditionalParameterEstimation",
                param_cmd,
                condition_func
            ))
        
        # M3C2 immer ausführen
        m3c2_runner = self.service_factory.get_m3c2_runner()
        commands.append(RunM3C2Command(m3c2_runner))
        
        # Bedingte Ausreißer-Erkennung
        if conditions.get('detect_outliers', True):
            outlier_strategy = self.service_factory.get_outlier_strategy(
                config.outlier_config.method,
                config.outlier_config.multiplier
            )
            commands.append(DetectOutliersCommand(outlier_strategy))
        
        # Speichern
        param_repo = self.service_factory.get_parameter_repository()
        commands.append(SaveResultsCommand(pc_repo, param_repo))
        
        # Bedingte Visualisierung
        if conditions.get('visualize', False):
            viz_service = self.service_factory.get_visualization_service()
            viz_cmd = GenerateVisualizationCommand(viz_service)
            condition_func = lambda ctx: ctx.get('config').output_format != 'json'
            commands.append(ConditionalCommand(
                "ConditionalVisualization",
                viz_cmd,
                condition_func
            ))
        
        pipeline_name = f"ConditionalPipeline_{config.cloud_pair.tag}"
        return Pipeline(commands, name=pipeline_name)
    
    def _create_parameter_estimation_command(self) -> Command:
        """Erstellt Parameter-Schätzungs-Command"""
        param_estimator = self.service_factory.get_param_estimator()
        sample_size = self.service_factory.get_config_value(
            'processing.sample_size', 10000
        )
        strategy = RadiusScanStrategy(sample_size=sample_size)
        return EstimateParametersCommand(param_estimator, strategy)


class LoadExistingResultsCommand(Command):
    """Lädt existierende Ergebnisse für Statistik-Berechnung"""
    
    def __init__(self, repository):
        super().__init__("LoadExistingResults")
        self.repository = repository
    
    def execute(self, context):
        self.log_execution(context)
        
        config = context.get('config')
        cloud_pair = config.cloud_pair
        
        # Baue Pfade zu existierenden Dateien
        folder_path = Path(config.cloud_pair.folder_id)
        prefix = cloud_pair.get_output_prefix(config.version)
        
        # Lade Distanzen
        dist_path = folder_path / f"{prefix}_m3c2_distances.txt"
        if not self.repository.exists(str(dist_path)):
            raise FileNotFoundError(f"Distance file not found: {dist_path}")
        
        distances = self.repository.load_distances(str(dist_path))
        
        # Lade Parameter wenn vorhanden
        from domain.entities import M3C2Parameters
        params_path = folder_path / f"{prefix}_m3c2_params.txt"
        
        if self.repository.exists(str(params_path)):
            # Verwende FileParameterRepository
            from infrastructure.repositories.file_point_cloud_repository import FileParameterRepository
            param_repo = FileParameterRepository(str(folder_path.parent))
            params_dict = param_repo.load_params(str(params_path))
            params = M3C2Parameters.from_dict(params_dict)
        else:
            # Default-Parameter
            params = M3C2Parameters(0.002, 0.004)
        
        # Erstelle M3C2Result
        from domain.entities import M3C2Result
        result = M3C2Result.from_arrays(
            distances=distances,
            uncertainties=np.zeros_like(distances),  # Dummy wenn nicht vorhanden
            parameters=params
        )
        
        context.set('m3c2_result', result)
        
        logger.info(f"Loaded existing results: {len(distances)} distances")
        
        return context
    
    def can_execute(self, context):
        return context.has('config')