# New_Architecture/application/factories/updated_pipeline_factory.py
"""Erweiterte Pipeline Factory mit Visualisierungs-Integration"""

import logging
from typing import List, Optional
from domain.commands.base import Command
from application.factories.service_factory import ServiceFactory

logger = logging.getLogger(__name__)


class UpdatedPipelineFactory:
    """Factory für Pipeline-Erstellung mit erweiterten Visualisierungen"""
    
    def __init__(self, service_factory: ServiceFactory):
        self.service_factory = service_factory
        
    def create_full_m3c2_pipeline(self, config) -> List[Command]:
        """Erstellt vollständige M3C2 Pipeline mit allen Visualisierungen"""
        commands = []
        
        # 1. Load Point Clouds
        from domain.commands.m3c2_commands import LoadPointCloudsCommandV2
        commands.append(LoadPointCloudsCommandV2(
            self.service_factory.get_point_cloud_repository()
        ))
        
        # 2. Estimate Parameters (wenn nicht vorhanden)
        if not config.use_existing_params:
            from domain.commands.m3c2_commands import EstimateParametersCommand
            commands.append(EstimateParametersCommand(
                self.service_factory.get_param_estimator()
            ))
        
        # 3. Run M3C2
        from domain.commands.m3c2_commands import RunM3C2Command
        commands.append(RunM3C2Command(
            self.service_factory.get_m3c2_runner()
        ))
        
        # 4. Save Coordinates with Distances (NEU!)
        from domain.commands.visualization_commands import SaveCoordinatesWithDistancesCommand
        commands.append(SaveCoordinatesWithDistancesCommand(
            self.service_factory.get_enhanced_visualization_service()
        ))
        
        # 5. Detect Outliers
        from domain.commands.m3c2_commands import DetectOutliersCommand
        strategy = self.service_factory.get_outlier_strategy(
            config.outlier_detection_method
        )
        commands.append(DetectOutliersCommand(strategy))
        
        # 6. Generate Outlier/Inlier PLYs (NEU!)
        from domain.commands.visualization_commands import GenerateOutlierInlierPLYsCommand
        commands.append(GenerateOutlierInlierPLYsCommand(
            self.service_factory.get_enhanced_visualization_service()
        ))
        
        # 7. Export PLY with Scalar Field (NEU!)
        from domain.commands.visualization_commands import ExportPLYWithScalarFieldCommand
        commands.append(ExportPLYWithScalarFieldCommand(
            self.service_factory.get_enhanced_visualization_service()
        ))
        
        # 8. Generate M3C2 Histograms (NEU!)
        from domain.commands.visualization_commands import GenerateM3C2HistogramCommand
        commands.append(GenerateM3C2HistogramCommand(
            self.service_factory.get_enhanced_visualization_service()
        ))
        
        # 9. Calculate Statistics
        from domain.commands.statistics_commands import CalculateStatisticsCommand
        commands.append(CalculateStatisticsCommand(
            self.service_factory.get_statistics_service()
        ))
        
        # 10. Generate Additional Plots
        from domain.commands.plot_commands import GeneratePlotsCommand
        plot_config = self.service_factory.get_config_value('plotting', {})
        commands.append(GeneratePlotsCommand(
            self.service_factory.get_plot_service(),
            plot_config
        ))
        
        # 11. Export Statistics
        from domain.commands.statistics_commands import ExportStatisticsCommand
        commands.append(ExportStatisticsCommand(
            self.service_factory.get_statistics_service(),
            config.output_format
        ))
        
        # 12. Create Visualization Report (NEU!)
        from domain.commands.visualization_commands import CreateVisualizationReportCommand
        commands.append(CreateVisualizationReportCommand())
        
        # 13. Save Final Results
        from domain.commands.m3c2_commands import SaveResultsCommandV2
        commands.append(SaveResultsCommandV2(
            self.service_factory.get_distance_repository()
        ))
        
        return commands
    
    def create_visualization_only_pipeline(self, config) -> List[Command]:
        """Pipeline nur für Visualisierung bestehender Ergebnisse"""
        commands = []
        
        # 1. Load existing results
        from domain.commands.m3c2_commands import LoadCloudCompareResultsCommand
        commands.append(LoadCloudCompareResultsCommand(
            self.service_factory.get_distance_repository()
        ))
        
        # 2. Convert TXT to PLY
        from domain.commands.visualization_commands import ConvertTXTtoPLYCommand
        commands.append(ConvertTXTtoPLYCommand(
            self.service_factory.get_enhanced_visualization_service()
        ))
        
        # 3. Generate Histograms
        from domain.commands.visualization_commands import GenerateM3C2HistogramCommand
        commands.append(GenerateM3C2HistogramCommand(
            self.service_factory.get_enhanced_visualization_service()
        ))
        
        # 4. Create Report
        from domain.commands.visualization_commands import CreateVisualizationReportCommand
        commands.append(CreateVisualizationReportCommand())
        
        return commands
    
    def create_batch_visualization_pipeline(self, configs: List) -> List[Command]:
        """Batch-Visualisierung für mehrere Konfigurationen"""
        from domain.commands.base import CompositeCommand
        
        commands = []
        
        # Verarbeite jede Konfiguration
        for i, config in enumerate(configs):
            viz_commands = []
            
            # Lade Daten
            from domain.commands.m3c2_commands import LoadCloudCompareResultsCommand
            viz_commands.append(LoadCloudCompareResultsCommand(
                self.service_factory.get_distance_repository()
            ))
            
            # Generiere alle Visualisierungen
            from domain.commands.visualization_commands import (
                ExportPLYWithScalarFieldCommand,
                GenerateOutlierInlierPLYsCommand,
                GenerateM3C2HistogramCommand
            )
            
            viz_commands.append(ExportPLYWithScalarFieldCommand(
                self.service_factory.get_enhanced_visualization_service()
            ))
            viz_commands.append(GenerateOutlierInlierPLYsCommand(
                self.service_factory.get_enhanced_visualization_service()
            ))
            viz_commands.append(GenerateM3C2HistogramCommand(
                self.service_factory.get_enhanced_visualization_service()
            ))
            
            # Composite für diese Konfiguration
            composite = CompositeCommand(
                f"VisualizationBatch_{i}_{config.cloud_pair.tag}",
                viz_commands
            )
            commands.append(composite)
        
        # Abschließender Report
        from domain.commands.visualization_commands import CreateVisualizationReportCommand
        commands.append(CreateVisualizationReportCommand())
        
        return commands