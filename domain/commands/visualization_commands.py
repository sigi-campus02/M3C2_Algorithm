# New_Architecture/domain/commands/visualization_commands.py
"""Visualization Commands für M3C2 Pipeline"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from domain.commands.base import Command, PipelineContext

logger = logging.getLogger(__name__)


class ExportPLYWithScalarFieldCommand(Command):
    """Exportiert PLY mit Distance als Scalar Field für CloudCompare"""
    
    def __init__(self, visualization_service, output_dir: Optional[Path] = None):
        super().__init__("ExportPLYWithScalarField")
        self.visualization_service = visualization_service
        self.output_dir = output_dir
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Exportiert PLY mit Scalar Field"""
        self.log_execution(context)
        
        # Hole benötigte Daten
        config = context.get('config')
        distances = context.get('m3c2_distances')
        cloud_pair = config.cloud_pair
        
        if not distances:
            logger.warning("No M3C2 distances found for PLY export")
            return context
        
        # Bestimme Output-Verzeichnis
        output_dir = self.output_dir or Path(config.output_dir) / config.folder_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Hole Koordinaten (entweder von M3C2 oder Original-Cloud)
        coordinates = distances.get('coordinates')
        if coordinates is None:
            # Fallback: Nutze mov cloud
            mov_cloud = context.get('cloud_mov')
            if hasattr(mov_cloud, 'cloud'):
                coordinates = np.asarray(mov_cloud.cloud)
            else:
                logger.error("No coordinates available for PLY export")
                return context
        
        # Exportiere verschiedene Versionen
        tag = cloud_pair.tag
        results = {}
        
        # 1. Mit allen Distanzen (inkl. NaN)
        all_distances = distances.get('with_outliers', distances.get('all', []))
        if len(all_distances) > 0:
            ply_all = output_dir / f"{config.process_python_CC}_{tag}_all_distances.ply"
            self.visualization_service.export_ply_with_scalar_field(
                coordinates,
                all_distances,
                ply_all,
                scalar_name="m3c2_distance"
            )
            results['all'] = ply_all
        
        # 2. Nur gültige Distanzen (ohne NaN)
        valid_mask = ~np.isnan(all_distances)
        if valid_mask.any():
            ply_valid = output_dir / f"{config.process_python_CC}_{tag}_valid_only.ply"
            self.visualization_service.export_ply_with_scalar_field(
                coordinates[valid_mask],
                all_distances[valid_mask],
                ply_valid,
                scalar_name="m3c2_distance"
            )
            results['valid'] = ply_valid
        
        context.set('ply_exports', results)
        logger.info(f"Exported {len(results)} PLY files with scalar fields")
        
        return context
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Prüft ob PLY Export möglich ist"""
        return context.has('m3c2_distances')


class GenerateOutlierInlierPLYsCommand(Command):
    """Generiert separate PLY-Dateien für Outlier und Inlier"""
    
    def __init__(self, visualization_service, output_dir: Optional[Path] = None):
        super().__init__("GenerateOutlierInlierPLYs")
        self.visualization_service = visualization_service
        self.output_dir = output_dir
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Generiert Outlier/Inlier PLYs"""
        self.log_execution(context)
        
        # Hole benötigte Daten
        config = context.get('config')
        distances = context.get('m3c2_distances')
        
        if not distances or 'outlier_mask' not in distances:
            logger.warning("No outlier information found for PLY generation")
            return context
        
        # Output-Verzeichnis
        output_dir = self.output_dir or Path(config.output_dir) / config.folder_id
        
        # Hole Daten
        coordinates = distances.get('coordinates')
        all_distances = distances.get('with_outliers')
        outlier_mask = distances.get('outlier_mask')
        
        if coordinates is None:
            mov_cloud = context.get('cloud_mov')
            if hasattr(mov_cloud, 'cloud'):
                coordinates = np.asarray(mov_cloud.cloud)
        
        # Generiere PLYs
        results = self.visualization_service.generate_outlier_inlier_plys(
            coordinates,
            all_distances,
            outlier_mask,
            output_dir,
            f"{config.process_python_CC}_{config.cloud_pair.tag}",
            config.outlier_detection_method
        )
        
        context.set('outlier_inlier_plys', results)
        logger.info(f"Generated outlier/inlier PLYs: {list(results.keys())}")
        
        return context
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Prüft ob Outlier/Inlier PLYs generiert werden können"""
        distances = context.get('m3c2_distances', {})
        return 'outlier_mask' in distances


class SaveCoordinatesWithDistancesCommand(Command):
    """Speichert Koordinaten mit Distanzen als TXT"""
    
    def __init__(self, visualization_service, output_dir: Optional[Path] = None):
        super().__init__("SaveCoordinatesWithDistances")
        self.visualization_service = visualization_service
        self.output_dir = output_dir
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Speichert x,y,z,distance TXT-Dateien"""
        self.log_execution(context)
        
        config = context.get('config')
        distances = context.get('m3c2_distances')
        
        if not distances:
            logger.warning("No distances to save with coordinates")
            return context
        
        # Output-Verzeichnis
        output_dir = self.output_dir or Path(config.output_dir) / config.folder_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Hole Koordinaten
        coordinates = distances.get('coordinates')
        if coordinates is None:
            mov_cloud = context.get('cloud_mov')
            if hasattr(mov_cloud, 'cloud'):
                coordinates = np.asarray(mov_cloud.cloud)
            else:
                logger.error("No coordinates available")
                return context
        
        tag = config.cloud_pair.tag
        results = {}
        
        # 1. Alle Distanzen mit Koordinaten
        all_distances = distances.get('with_outliers', distances.get('all', []))
        if len(all_distances) > 0:
            txt_all = output_dir / f"python_{tag}_m3c2_distances_coordinates.txt"
            self.visualization_service.save_coordinates_with_distances(
                coordinates, all_distances, txt_all
            )
            results['all'] = txt_all
        
        # 2. Nur Inlier
        if 'outlier_mask' in distances:
            outlier_mask = distances['outlier_mask']
            inlier_mask = ~outlier_mask
            
            if inlier_mask.any():
                txt_inlier = output_dir / f"python_{tag}_m3c2_distances_coordinates_inlier_{config.outlier_detection_method}.txt"
                self.visualization_service.save_coordinates_with_distances(
                    coordinates[inlier_mask],
                    all_distances[inlier_mask],
                    txt_inlier
                )
                results['inlier'] = txt_inlier
            
            # 3. Nur Outlier
            if outlier_mask.any():
                txt_outlier = output_dir / f"python_{tag}_m3c2_distances_coordinates_outlier_{config.outlier_detection_method}.txt"
                self.visualization_service.save_coordinates_with_distances(
                    coordinates[outlier_mask],
                    all_distances[outlier_mask],
                    txt_outlier
                )
                results['outlier'] = txt_outlier
        
        context.set('coordinate_distance_files', results)
        logger.info(f"Saved {len(results)} coordinate-distance files")
        
        return context
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Prüft ob Koordinaten mit Distanzen gespeichert werden können"""
        return context.has('m3c2_distances')


class GenerateM3C2HistogramCommand(Command):
    """Generiert M3C2-spezifische Histogramme"""
    
    def __init__(self, visualization_service, output_dir: Optional[Path] = None):
        super().__init__("GenerateM3C2Histogram")
        self.visualization_service = visualization_service
        self.output_dir = output_dir
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Generiert M3C2 Histogramme"""
        self.log_execution(context)
        
        config = context.get('config')
        distances = context.get('m3c2_distances')
        
        if not distances:
            logger.warning("No distances for histogram generation")
            return context
        
        # Output-Verzeichnis
        output_dir = self.output_dir or Path(config.output_dir) / config.folder_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tag = config.cloud_pair.tag
        results = {}
        
        # 1. Histogramm mit allen Distanzen
        all_distances = distances.get('with_outliers', distances.get('all', []))
        if len(all_distances) > 0:
            hist_all = output_dir / f"{config.process_python_CC}_{tag}_histogram.png"
            self.visualization_service.create_m3c2_histogram(
                all_distances,
                hist_all,
                title=f"M3C2 Distances - {tag}",
                bins=256,
                show_statistics=True
            )
            results['all'] = hist_all
        
        # 2. Vergleichs-Histogramm wenn Outlier vorhanden
        if 'outlier_mask' in distances:
            outlier_mask = distances['outlier_mask']
            inlier_mask = ~outlier_mask
            
            comparison_data = {}
            if inlier_mask.any():
                comparison_data['Inliers'] = all_distances[inlier_mask]
            if outlier_mask.any():
                comparison_data['Outliers'] = all_distances[outlier_mask]
            
            if len(comparison_data) > 1:
                hist_comparison = output_dir / f"{config.process_python_CC}_{tag}_histogram_comparison.png"
                self.visualization_service.create_comparison_histogram(
                    comparison_data,
                    hist_comparison,
                    title=f"Inlier vs Outlier - {tag}"
                )
                results['comparison'] = hist_comparison
        
        context.set('histogram_files', results)
        logger.info(f"Generated {len(results)} histograms")
        
        return context
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Prüft ob Histogramme generiert werden können"""
        return context.has('m3c2_distances')


class CreateVisualizationReportCommand(Command):
    """Erstellt einen Visualisierungs-Report mit allen generierten Outputs"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        super().__init__("CreateVisualizationReport")
        self.output_dir = output_dir
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Erstellt Visualisierungs-Report"""
        self.log_execution(context)
        
        config = context.get('config')
        output_dir = self.output_dir or Path(config.output_dir) / config.folder_id
        
        # Sammle alle generierten Visualisierungen
        report = {
            'tag': config.cloud_pair.tag,
            'folder': config.folder_id,
            'ply_exports': context.get('ply_exports', {}),
            'outlier_inlier_plys': context.get('outlier_inlier_plys', {}),
            'coordinate_files': context.get('coordinate_distance_files', {}),
            'histograms': context.get('histogram_files', {}),
            'plots': context.get('plot_results', {})
        }
        
        # Schreibe Report als JSON
        import json
        report_path = output_dir / f"{config.cloud_pair.tag}_visualization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        context.set('visualization_report', report_path)
        
        # Log Summary
        total_files = sum(len(v) if isinstance(v, dict) else 0 for v in report.values())
        logger.info(f"Visualization report created: {report_path} ({total_files} files)")
        
        return context
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Report kann immer erstellt werden"""
        return True


class ConvertTXTtoPLYCommand(Command):
    """Konvertiert TXT-Dateien mit Distanzen zu PLY"""
    
    def __init__(self, visualization_service, input_pattern: str = "*_distances_coordinates*.txt"):
        super().__init__("ConvertTXTtoPLY")
        self.visualization_service = visualization_service
        self.input_pattern = input_pattern
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Konvertiert TXT zu PLY"""
        self.log_execution(context)
        
        config = context.get('config')
        data_dir = Path(config.output_dir) / config.folder_id
        
        if not data_dir.exists():
            logger.warning(f"Directory not found: {data_dir}")
            return context
        
        # Finde alle passenden TXT-Dateien
        txt_files = list(data_dir.glob(self.input_pattern))
        
        converted = []
        for txt_path in txt_files:
            try:
                ply_path = txt_path.with_suffix('.ply')
                self.visualization_service.txt_to_ply_with_distance_color(
                    txt_path,
                    ply_path,
                    scalar_name="distance"
                )
                converted.append(ply_path)
            except Exception as e:
                logger.error(f"Failed to convert {txt_path}: {e}")
        
        context.set('converted_plys', converted)
        logger.info(f"Converted {len(converted)} TXT files to PLY")
        
        return context
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Kann ausgeführt werden wenn Output-Verzeichnis existiert"""
        config = context.get('config')
        return Path(config.output_dir).exists()