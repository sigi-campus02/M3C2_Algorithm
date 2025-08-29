# New_Architecture/domain/commands/statistics_commands.py
"""Commands für Statistik-Berechnungen"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import pandas as pd
from domain.commands.base import Command, PipelineContext
from domain.entities import CloudPair, Statistics

logger = logging.getLogger(__name__)


class CalculateStatisticsCommand(Command):
    """Berechnet Statistiken für M3C2 Distanzen"""
    
    def __init__(self, statistics_service):
        super().__init__("CalculateStatistics")
        self.statistics_service = statistics_service
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Berechnet Statistiken"""
        self.log_execution(context)
        
        config = context.get('config')
        distances = context.get('m3c2_distances', {})
        
        if not distances:
            logger.warning("No distances found for statistics calculation")
            return context
        
        # Berechne Statistiken für WITH und INLIER
        stats_with = self._calculate_stats(
            distances.get('with_outliers', np.array([])),
            "with_outliers"
        )
        
        stats_inlier = self._calculate_stats(
            distances.get('inliers', np.array([])),
            "inliers"
        )
        
        # Kombiniere Statistiken
        statistics = Statistics(
            cloud_pair=config.cloud_pair,
            with_outliers=stats_with,
            inliers_only=stats_inlier,
            outlier_count=len(distances.get('outliers', [])),
            total_count=len(distances.get('with_outliers', []))
        )
        
        context.set('statistics', statistics)
        context.set('stats_dict', {
            'with_outliers': stats_with,
            'inliers': stats_inlier,
            'outlier_percentage': statistics.outlier_percentage
        })
        
        logger.info(f"Calculated statistics: {statistics.outlier_percentage:.1f}% outliers")
        
        return context
    
    def _calculate_stats(self, data: np.ndarray, label: str) -> Dict[str, float]:
        """Berechnet detaillierte Statistiken"""
        if len(data) == 0:
            return self._empty_stats()
        
        return {
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'q1': float(np.percentile(data, 25)),
            'q3': float(np.percentile(data, 75)),
            'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
            'rmse': float(np.sqrt(np.mean(data**2))),
            'mae': float(np.mean(np.abs(data))),
            'count': len(data),
            'label': label
        }
    
    def _empty_stats(self) -> Dict[str, float]:
        """Gibt leere Statistiken zurück"""
        return {
            'mean': 0.0, 'median': 0.0, 'std': 0.0,
            'min': 0.0, 'max': 0.0, 'q1': 0.0, 'q3': 0.0,
            'iqr': 0.0, 'rmse': 0.0, 'mae': 0.0, 'count': 0
        }
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Prüft ob Statistiken berechnet werden können"""
        return context.has('m3c2_distances') or context.has('distances')


class CalculateBatchStatisticsCommand(Command):
    """Berechnet aggregierte Statistiken für Batch-Verarbeitung"""
    
    def __init__(self, statistics_service):
        super().__init__("CalculateBatchStatistics")
        self.statistics_service = statistics_service
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Berechnet Batch-Statistiken"""
        self.log_execution(context)
        
        batch_results = context.get('batch_results', [])
        if not batch_results:
            logger.warning("No batch results found for statistics")
            return context
        
        # Sammle Statistiken pro Index und Case
        grouped_stats = self._group_statistics_by_index(batch_results)
        
        # Berechne aggregierte Statistiken
        aggregated = self._aggregate_statistics(grouped_stats)
        
        # Erstelle DataFrame für Export
        df_stats = self._create_statistics_dataframe(grouped_stats, aggregated)
        
        context.set('batch_statistics', grouped_stats)
        context.set('aggregated_statistics', aggregated)
        context.set('statistics_dataframe', df_stats)
        
        logger.info(f"Calculated batch statistics for {len(grouped_stats)} groups")
        
        return context
    
    def _group_statistics_by_index(self, batch_results: List[Dict]) -> Dict[str, Dict]:
        """Gruppiert Statistiken nach Index"""
        grouped = {}
        
        for result in batch_results:
            if 'statistics' not in result:
                continue
            
            stats = result['statistics']
            index = result.get('index', 'unknown')
            case = result.get('case', 'default')
            
            key = f"part_{index}_{case}"
            grouped[key] = {
                'index': index,
                'case': case,
                'with_outliers': stats.get('with_outliers', {}),
                'inliers': stats.get('inliers', {}),
                'outlier_percentage': stats.get('outlier_percentage', 0)
            }
        
        return grouped
    
    def _aggregate_statistics(self, grouped_stats: Dict) -> Dict:
        """Berechnet aggregierte Statistiken über alle Gruppen"""
        all_means_with = []
        all_means_inlier = []
        all_stds_with = []
        all_stds_inlier = []
        all_outlier_percentages = []
        
        for key, stats in grouped_stats.items():
            if stats['with_outliers']:
                all_means_with.append(stats['with_outliers'].get('mean', 0))
                all_stds_with.append(stats['with_outliers'].get('std', 0))
            
            if stats['inliers']:
                all_means_inlier.append(stats['inliers'].get('mean', 0))
                all_stds_inlier.append(stats['inliers'].get('std', 0))
            
            all_outlier_percentages.append(stats['outlier_percentage'])
        
        return {
            'overall_mean_with': np.mean(all_means_with) if all_means_with else 0,
            'overall_std_with': np.mean(all_stds_with) if all_stds_with else 0,
            'overall_mean_inlier': np.mean(all_means_inlier) if all_means_inlier else 0,
            'overall_std_inlier': np.mean(all_stds_inlier) if all_stds_inlier else 0,
            'mean_outlier_percentage': np.mean(all_outlier_percentages) if all_outlier_percentages else 0,
            'total_groups': len(grouped_stats)
        }
    
    def _create_statistics_dataframe(self, grouped_stats: Dict, aggregated: Dict) -> pd.DataFrame:
        """Erstellt DataFrame für Export"""
        rows = []
        
        # Einzelne Gruppen
        for key, stats in grouped_stats.items():
            row = {
                'Group': key,
                'Index': stats['index'],
                'Case': stats['case'],
                'Mean_WITH': stats['with_outliers'].get('mean', 0) if stats['with_outliers'] else 0,
                'Std_WITH': stats['with_outliers'].get('std', 0) if stats['with_outliers'] else 0,
                'Mean_INLIER': stats['inliers'].get('mean', 0) if stats['inliers'] else 0,
                'Std_INLIER': stats['inliers'].get('std', 0) if stats['inliers'] else 0,
                'Outlier_%': stats['outlier_percentage']
            }
            rows.append(row)
        
        # Aggregierte Zeile
        rows.append({
            'Group': 'OVERALL',
            'Index': 'ALL',
            'Case': 'AGGREGATED',
            'Mean_WITH': aggregated['overall_mean_with'],
            'Std_WITH': aggregated['overall_std_with'],
            'Mean_INLIER': aggregated['overall_mean_inlier'],
            'Std_INLIER': aggregated['overall_std_inlier'],
            'Outlier_%': aggregated['mean_outlier_percentage']
        })
        
        return pd.DataFrame(rows)
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Prüft ob Batch-Statistiken berechnet werden können"""
        return context.has('batch_results')


class CalculateSingleCloudStatisticsCommand(Command):
    """Berechnet Statistiken für einzelne Punktwolken"""
    
    def __init__(self, statistics_service):
        super().__init__("CalculateSingleCloudStatistics")
        self.statistics_service = statistics_service
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Berechnet Einzelwolken-Statistiken"""
        self.log_execution(context)
        
        config = context.get('config')
        cloud_mov = context.get('cloud_mov')
        cloud_ref = context.get('cloud_ref')
        
        if cloud_mov is None or cloud_ref is None:
            logger.warning("Point clouds not loaded for statistics")
            return context
        
        # Berechne Cloud-Statistiken
        stats_mov = self._calculate_cloud_stats(cloud_mov, "moving")
        stats_ref = self._calculate_cloud_stats(cloud_ref, "reference")
        
        # Berechne Vergleichs-Metriken
        comparison = self._calculate_comparison_metrics(cloud_mov, cloud_ref)
        
        cloud_statistics = {
            'moving_cloud': stats_mov,
            'reference_cloud': stats_ref,
            'comparison': comparison,
            'cloud_pair': config.cloud_pair.tag
        }
        
        context.set('cloud_statistics', cloud_statistics)
        logger.info("Calculated single cloud statistics")
        
        return context
    
    def _calculate_cloud_stats(self, cloud: np.ndarray, label: str) -> Dict:
        """Berechnet Statistiken für eine einzelne Punktwolke"""
        if len(cloud) == 0:
            return {'label': label, 'point_count': 0}
        
        # Berechne Bounding Box
        bbox_min = np.min(cloud, axis=0)
        bbox_max = np.max(cloud, axis=0)
        bbox_size = bbox_max - bbox_min
        
        # Berechne Zentrum und Ausdehnung
        centroid = np.mean(cloud, axis=0)
        
        # Punkt-Dichte Schätzung (vereinfacht)
        if len(cloud) > 1:
            # K-nearest neighbor für Dichte
            from scipy.spatial import KDTree
            tree = KDTree(cloud)
            distances, _ = tree.query(cloud[:min(1000, len(cloud))], k=2)
            avg_nn_distance = np.mean(distances[:, 1])
        else:
            avg_nn_distance = 0
        
        return {
            'label': label,
            'point_count': len(cloud),
            'bbox_min': bbox_min.tolist(),
            'bbox_max': bbox_max.tolist(),
            'bbox_size': bbox_size.tolist(),
            'volume': float(np.prod(bbox_size)),
            'centroid': centroid.tolist(),
            'avg_nn_distance': float(avg_nn_distance),
            'estimated_density': 1.0 / (avg_nn_distance ** 3) if avg_nn_distance > 0 else 0
        }
    
    def _calculate_comparison_metrics(self, cloud1: np.ndarray, cloud2: np.ndarray) -> Dict:
        """Berechnet Vergleichs-Metriken zwischen zwei Wolken"""
        if len(cloud1) == 0 or len(cloud2) == 0:
            return {'overlap_ratio': 0, 'hausdorff_distance': 0}
        
        # Vereinfachte Überlappungs-Berechnung
        bbox1_min = np.min(cloud1, axis=0)
        bbox1_max = np.max(cloud1, axis=0)
        bbox2_min = np.min(cloud2, axis=0)
        bbox2_max = np.max(cloud2, axis=0)
        
        # Intersection
        inter_min = np.maximum(bbox1_min, bbox2_min)
        inter_max = np.minimum(bbox1_max, bbox2_max)
        
        if np.all(inter_max > inter_min):
            inter_volume = np.prod(inter_max - inter_min)
            union_volume = (np.prod(bbox1_max - bbox1_min) + 
                          np.prod(bbox2_max - bbox2_min) - inter_volume)
            overlap_ratio = inter_volume / union_volume if union_volume > 0 else 0
        else:
            overlap_ratio = 0
        
        # Vereinfachte Hausdorff-Distanz (Sample-basiert für Performance)
        sample_size = min(1000, len(cloud1), len(cloud2))
        if sample_size > 0:
            indices1 = np.random.choice(len(cloud1), sample_size, replace=False)
            indices2 = np.random.choice(len(cloud2), sample_size, replace=False)
            
            from scipy.spatial.distance import directed_hausdorff
            hausdorff_dist = max(
                directed_hausdorff(cloud1[indices1], cloud2[indices2])[0],
                directed_hausdorff(cloud2[indices2], cloud1[indices1])[0]
            )
        else:
            hausdorff_dist = 0
        
        return {
            'overlap_ratio': float(overlap_ratio),
            'hausdorff_distance': float(hausdorff_dist),
            'point_count_ratio': len(cloud1) / len(cloud2) if len(cloud2) > 0 else 0
        }
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Prüft ob Cloud-Statistiken berechnet werden können"""
        return context.has('cloud_mov') and context.has('cloud_ref')


class ExportStatisticsCommand(Command):
    """Exportiert Statistiken in verschiedene Formate"""
    
    def __init__(self, statistics_service, export_format: str = 'excel'):
        super().__init__("ExportStatistics")
        self.statistics_service = statistics_service
        self.export_format = export_format.lower()
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Exportiert Statistiken"""
        self.log_execution(context)
        
        config = context.get('config')
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        export_paths = []
        
        # Export einzelne Statistiken
        if context.has('statistics'):
            stats = context.get('statistics')
            path = self._export_single_statistics(stats, output_dir, config.cloud_pair.tag)
            export_paths.append(path)
        
        # Export Batch-Statistiken
        if context.has('statistics_dataframe'):
            df = context.get('statistics_dataframe')
            path = self._export_batch_statistics(df, output_dir, config.project_name)
            export_paths.append(path)
        
        # Export Cloud-Statistiken
        if context.has('cloud_statistics'):
            cloud_stats = context.get('cloud_statistics')
            path = self._export_cloud_statistics(cloud_stats, output_dir, config.cloud_pair.tag)
            export_paths.append(path)
        
        context.set('statistics_export_paths', export_paths)
        logger.info(f"Exported statistics to {len(export_paths)} files")
        
        return context
    
    def _export_single_statistics(self, stats: Statistics, output_dir: Path, tag: str) -> Path:
        """Exportiert einzelne Statistiken"""
        if self.export_format == 'excel':
            output_path = output_dir / f"{tag}_statistics.xlsx"
            self.statistics_service.export_to_excel(stats, output_path)
        elif self.export_format == 'csv':
            output_path = output_dir / f"{tag}_statistics.csv"
            self.statistics_service.export_to_csv(stats, output_path)
        elif self.export_format == 'json':
            output_path = output_dir / f"{tag}_statistics.json"
            self.statistics_service.export_to_json(stats, output_path)
        else:
            raise ValueError(f"Unsupported export format: {self.export_format}")
        
        return output_path
    
    def _export_batch_statistics(self, df: pd.DataFrame, output_dir: Path, project: str) -> Path:
        """Exportiert Batch-Statistiken"""
        if self.export_format == 'excel':
            output_path = output_dir / f"{project}_batch_statistics.xlsx"
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Statistics', index=False)
                
                # Formatierung
                worksheet = writer.sheets['Statistics']
                for column in worksheet.columns:
                    max_length = max(len(str(cell.value)) for cell in column)
                    worksheet.column_dimensions[column[0].column_letter].width = min(max_length + 2, 30)
        else:
            output_path = output_dir / f"{project}_batch_statistics.csv"
            df.to_csv(output_path, index=False)
        
        return output_path
    
    def _export_cloud_statistics(self, cloud_stats: Dict, output_dir: Path, tag: str) -> Path:
        """Exportiert Cloud-Statistiken"""
        output_path = output_dir / f"{tag}_cloud_statistics.json"
        
        import json
        with open(output_path, 'w') as f:
            json.dump(cloud_stats, f, indent=2)
        
        return output_path
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Prüft ob Statistiken exportiert werden können"""
        return (context.has('statistics') or 
                context.has('statistics_dataframe') or 
                context.has('cloud_statistics'))