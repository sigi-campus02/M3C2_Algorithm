# New_Architecture/domain/commands/m3c2_commands.py
"""M3C2 Pipeline Commands - Clean Implementation"""

import logging
import numpy as np
from typing import Optional
from pathlib import Path

from domain.commands.base import Command, PipelineContext
from domain.entities import M3C2Parameters, M3C2Result
from infrastructure.repositories.enhanced_point_cloud_repository import EnhancedPointCloudRepository
from infrastructure.repositories.distance_repository import DistanceRepository
from infrastructure.repositories.file_point_cloud_repository import FileParameterRepository
from orchestration.m3c2_runner import M3C2Runner

logger = logging.getLogger(__name__)


class LoadPointCloudsCommand(Command):
    """Lädt Punktwolken mit Multi-Format Support"""
    
    def __init__(self, pc_repository: EnhancedPointCloudRepository):
        super().__init__("LoadPointClouds")
        self.repository = pc_repository
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        self.log_execution(context)
        
        config = context.get('config')
        cloud_pair = config.cloud_pair
        
        # Finde und lade Moving Cloud
        mov_path = self._find_cloud_file(cloud_pair.folder_id, cloud_pair.moving_cloud)
        if not mov_path:
            raise FileNotFoundError(f"Moving cloud not found: {cloud_pair.moving_cloud}")
        
        # Finde und lade Reference Cloud
        ref_path = self._find_cloud_file(cloud_pair.folder_id, cloud_pair.reference_cloud)
        if not ref_path:
            raise FileNotFoundError(f"Reference cloud not found: {cloud_pair.reference_cloud}")
        
        try:
            # Lade mit automatischer Format-Erkennung
            mov_cloud = self.repository.load_point_cloud(mov_path)
            ref_cloud = self.repository.load_point_cloud(ref_path)
            
            logger.info(
                f"Loaded clouds - Moving: {mov_cloud.cloud.shape}, "
                f"Reference: {ref_cloud.cloud.shape}"
            )
            
            # Speichere im Kontext
            context.set('moving_cloud', mov_cloud)
            context.set('reference_cloud', ref_cloud)
            
            # Bestimme Corepoints
            corepoints = self._determine_corepoints(mov_cloud, ref_cloud, config)
            context.set('corepoints', corepoints)
            
        except Exception as e:
            self.handle_error(e, context)
            raise
        
        return context
    
    def _find_cloud_file(self, folder: str, basename: str) -> Optional[str]:
        """Sucht nach Punktwolken-Datei in verschiedenen Formaten"""
        extensions = ['.ply', '.xyz', '.las', '.laz', '.obj', '.gpc', '.txt']
        
        for ext in extensions:
            test_path = f"{folder}/{basename}{ext}"
            if self.repository.exists(test_path):
                return test_path
        
        return None
    
    def _determine_corepoints(self, mov_cloud, ref_cloud, config) -> np.ndarray:
        """Bestimmt Corepoints basierend auf Konfiguration"""
        if config.mov_as_corepoints:
            corepoints = np.asarray(mov_cloud.cloud)
        else:
            corepoints = np.asarray(ref_cloud.cloud)
        
        # Subsampling wenn gewünscht
        if config.use_subsampled_corepoints > 1:
            step = config.use_subsampled_corepoints
            corepoints = corepoints[::step]
            logger.info(f"Subsampled corepoints by factor {step}: {len(corepoints)} points")
        
        return corepoints
    
    def can_execute(self, context: PipelineContext) -> bool:
        return context.has('config')


class EstimateParametersCommand(Command):
    """Schätzt M3C2-Parameter automatisch"""
    
    def __init__(self, param_estimator, strategy):
        super().__init__("EstimateParameters")
        self.param_estimator = param_estimator
        self.strategy = strategy
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        self.log_execution(context)
        
        config = context.get('config')
        
        # Prüfe ob Parameter bereits gesetzt sind
        if config.m3c2_params:
            logger.info("Using provided M3C2 parameters")
            context.set('m3c2_params', config.m3c2_params)
            return context
        
        # Schätze Parameter
        corepoints = context.get('corepoints')
        
        try:
            # Berechne durchschnittlichen Punktabstand
            avg_spacing = self.param_estimator.estimate_min_spacing(corepoints)
            logger.info(f"Average point spacing: {avg_spacing:.6f}")
            
            # Scanne verschiedene Skalen
            scans = self.param_estimator.scan_scales(corepoints, self.strategy, avg_spacing)
            logger.info(f"Evaluated {len(scans)} scales")
            
            # Wähle beste Parameter
            normal_scale, search_scale = self.param_estimator.select_scales(scans)
            
            params = M3C2Parameters(normal_scale, search_scale)
            context.set('m3c2_params', params)
            
            logger.info(f"Estimated parameters: normal={normal_scale:.6f}, search={search_scale:.6f}")
            
        except Exception as e:
            self.handle_error(e, context)
            raise
        
        return context
    
    def can_execute(self, context: PipelineContext) -> bool:
        return context.has('corepoints')


class RunM3C2Command(Command):
    """Führt M3C2-Algorithmus aus"""
    
    def __init__(self, m3c2_runner: M3C2Runner):
        super().__init__("RunM3C2")
        self.m3c2_runner = m3c2_runner
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        self.log_execution(context)
        
        mov = context.get('moving_cloud')
        ref = context.get('reference_cloud')
        corepoints = context.get('corepoints')
        params = context.get('m3c2_params')
        
        try:
            # Führe M3C2 aus
            distances, uncertainties = self.m3c2_runner.run(
                mov, ref, corepoints,
                params.normal_scale,
                params.search_scale
            )
            
            # Erstelle Result-Objekt
            result = M3C2Result.from_arrays(distances, uncertainties, params)
            context.set('m3c2_result', result)
            
            logger.info(
                f"M3C2 completed: {result.valid_count} valid points, "
                f"{result.nan_percentage:.1f}% NaN"
            )
            
        except Exception as e:
            self.handle_error(e, context)
            raise
        
        return context
    
    def can_execute(self, context: PipelineContext) -> bool:
        required = ['moving_cloud', 'reference_cloud', 'corepoints', 'm3c2_params']
        return all(context.has(key) for key in required)


class DetectOutliersCommand(Command):
    """Erkennt und markiert Ausreißer"""
    
    def __init__(self, outlier_strategy):
        super().__init__("DetectOutliers")
        self.outlier_strategy = outlier_strategy
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        self.log_execution(context)
        
        result = context.get('m3c2_result')
        distances = result.distances
        
        try:
            # Erkenne Ausreißer
            outliers, threshold = self.outlier_strategy.detect(distances)
            
            # Berechne Statistiken
            stats = self.outlier_strategy.get_statistics(distances, outliers)
            
            # Speichere im Kontext
            context.set('outliers', outliers)
            context.set('outlier_threshold', threshold)
            context.set('outlier_stats', stats)
            
            # Erstelle Inlier-Array
            inliers = ~outliers & ~np.isnan(distances)
            context.set('inliers', inliers)
            
            logger.info(
                f"Outlier detection: {stats['outlier_count']} outliers "
                f"({stats['outlier_percentage']:.1f}%), threshold={threshold:.6f}"
            )
            
        except Exception as e:
            self.handle_error(e, context)
            raise
        
        return context
    
    def can_execute(self, context: PipelineContext) -> bool:
        return context.has('m3c2_result')


class SaveResultsCommand(Command):
    """Speichert Ergebnisse mit Repository"""
    
    def __init__(
        self,
        dist_repository: DistanceRepository,
        param_repository: FileParameterRepository
    ):
        super().__init__("SaveResults")
        self.dist_repo = dist_repository
        self.param_repo = param_repository
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        self.log_execution(context)
        
        config = context.get('config')
        result = context.get('m3c2_result')
        mov_cloud = context.get('moving_cloud')
        
        # Baue Output-Pfade
        output_base = Path(config.cloud_pair.folder_id)
        prefix = config.cloud_pair.get_output_prefix(config.version)
        
        try:
            # Speichere Distanzen
            dist_path = f"{output_base}/{prefix}_m3c2_distances.txt"
            self.dist_repo.save_distances(result.distances, dist_path)
            
            # Speichere Unsicherheiten
            uncert_path = f"{output_base}/{prefix}_m3c2_uncertainties.txt"
            self.dist_repo.save_distances(result.uncertainties, uncert_path)
            
            # Speichere Distanzen mit Koordinaten
            coords = np.asarray(mov_cloud.cloud)
            coord_dist_path = f"{output_base}/{prefix}_m3c2_distances_coordinates.txt"
            self.dist_repo.save_distances_with_coordinates(
                coords, result.distances, coord_dist_path
            )
            
            # Speichere Parameter
            params_path = f"{output_base}/{prefix}_m3c2_params.txt"
            self.param_repo.save_params(
                result.parameters_used.to_dict(),
                params_path
            )
            
            # Speichere Outlier/Inlier wenn vorhanden
            if context.has('outliers'):
                outliers = context.get('outliers')
                base_path = f"{output_base}/{prefix}_m3c2_distances_coordinates"
                
                outlier_path, inlier_path = self.dist_repo.split_by_outliers(
                    coords,
                    result.distances,
                    outliers,
                    base_path,
                    method=config.outlier_config.method.value
                )
                
                context.set('outlier_path', outlier_path)
                context.set('inlier_path', inlier_path)
            
            logger.info(f"Results saved to {output_base}")
            
        except Exception as e:
            self.handle_error(e, context)
            raise
        
        return context
    
    def can_execute(self, context: PipelineContext) -> bool:
        return context.has('m3c2_result') and context.has('config')


class ComputeStatisticsCommand(Command):
    """Berechnet statistische Kennzahlen"""
    
    def __init__(self, statistics_service):
        super().__init__("ComputeStatistics")
        self.statistics_service = statistics_service
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        self.log_execution(context)
        
        result = context.get('m3c2_result')
        config = context.get('config')
        
        try:
            # Berechne Statistiken
            stats = self.statistics_service.compute_statistics(
                distances=result.distances,
                params=result.parameters_used,
                outlier_config=config.outlier_config
            )
            
            # Füge Metadaten hinzu
            stats['folder_id'] = config.cloud_pair.folder_id
            stats['cloud_pair'] = config.cloud_pair.tag
            stats['comparison_case'] = config.cloud_pair.comparison_case.value
            
            context.set('statistics', stats)
            
            logger.info("Statistics computed successfully")
            
        except Exception as e:
            self.handle_error(e, context)
            raise
        
        return context
    
    def can_execute(self, context: PipelineContext) -> bool:
        return context.has('m3c2_result')


class GenerateVisualizationCommand(Command):
    """Generiert Visualisierungen"""
    
    def __init__(self, visualization_service):
        super().__init__("GenerateVisualization")
        self.visualization_service = visualization_service
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        self.log_execution(context)
        
        config = context.get('config')
        result = context.get('m3c2_result')
        mov_cloud = context.get('moving_cloud')
        
        # Baue Output-Pfade
        output_base = Path(config.cloud_pair.folder_id)
        prefix = config.cloud_pair.get_output_prefix(config.version)
        
        try:
            # Generiere Histogram
            hist_path = output_base / f"{prefix}_histogram.png"
            self.visualization_service.create_histogram(
                result.distances,
                str(hist_path)
            )
            
            # Generiere farbkodierte Punktwolke
            ply_path = output_base / f"{prefix}_colored.ply"
            self.visualization_service.create_colored_cloud(
                mov_cloud,
                result.distances,
                str(ply_path)
            )
            
            logger.info(f"Visualizations saved to {output_base}")
            
        except Exception as e:
            # Visualisierung ist nicht kritisch
            logger.warning(f"Visualization failed: {e}")
            context.add_error(f"Visualization failed: {e}")
        
        return context
    
    def can_execute(self, context: PipelineContext) -> bool:
        return context.has('m3c2_result') and context.has('moving_cloud')