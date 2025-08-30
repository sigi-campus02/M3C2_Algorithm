# New_Architecture/application/services/service_integration.py
"""Integration der modularisierten Services"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np

from application.services.modular_services import ModularPlotService
from application.services.modular_statistics_service import ModularStatisticsService
from application.services.visualization_service import VisualizationService
from application.services.export_service import ExportService
from application.services.report_service import ReportService

logger = logging.getLogger(__name__)


class IntegratedServiceManager:
    """Manager für alle Services mit einheitlicher API"""
    
    def __init__(self, service_factory):
        self.factory = service_factory
        
        # Initialize services
        self.plot_service = ModularPlotService(
            repository=self.factory.get_point_cloud_repository(),
            theme='default'
        )
        
        self.statistics_service = ModularStatisticsService(
            repository=self.factory.get_statistics_repository(),
            default_strategies=['basic', 'advanced', 'distance']
        )
        
        self.visualization_service = VisualizationService(
            repository=self.factory.get_point_cloud_repository()
        )
        
        self.export_service = ExportService()
        self.report_service = ReportService()
        
        logger.info("Initialized IntegratedServiceManager")
    
    def process_m3c2_results(
        self,
        distances: np.ndarray,
        coordinates: np.ndarray,
        output_dir: Path,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verarbeitet M3C2 Ergebnisse mit allen Services"""
        results = {}
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Calculate Statistics
        logger.info("Calculating statistics...")
        statistics = self.statistics_service.calculate_statistics(
            distances,
            strategies=['basic', 'advanced', 'distance', 'normality'],
            cache_key=f"m3c2_{config.get('tag', 'default')}"
        )
        results['statistics'] = statistics
        
        # 2. Generate Plots
        logger.info("Generating plots...")
        plot_configs = [
            {'type': 'histogram', 'data': distances, 'filename': 'histogram.png',
             'title': 'Distance Distribution', 'bins': 50},
            {'type': 'gaussian', 'data': distances, 'filename': 'gaussian_fit.png',
             'title': 'Gaussian Fit'},
            {'type': 'boxplot', 'data': {'Distances': distances}, 
             'filename': 'boxplot.png', 'title': 'Distance Box Plot'},
            {'type': 'qq', 'data': distances, 'filename': 'qq_plot.png',
             'title': 'Q-Q Plot'}
        ]
        
        plots = self.plot_service.create_multiple_plots(
            plot_configs, 
            output_dir / 'plots'
        )
        results['plots'] = plots
        
        # 3. Create 3D Visualization
        logger.info("Creating 3D visualization...")
        viz_path = output_dir / 'visualization.html'
        viz_fig = self.visualization_service.create_distance_visualization(
            points=coordinates,
            distances=distances,
            output_path=viz_path,
            title='M3C2 Distance Visualization'
        )
        results['visualization'] = viz_path
        
        # 4. Export Results
        logger.info("Exporting results...")
        export_data = {
            'Statistics': self._flatten_statistics(statistics),
            'Distances': pd.DataFrame({
                'X': coordinates[:, 0],
                'Y': coordinates[:, 1],
                'Z': coordinates[:, 2],
                'Distance': distances
            })
        }
        
        excel_path = output_dir / 'results.xlsx'
        self.export_service.export_to_excel(
            export_data,
            excel_path,
            metadata={'project': config.get('project', 'M3C2 Analysis')}
        )
        results['excel'] = excel_path
        
        # 5. Generate Report
        logger.info("Generating report...")
        report_path = output_dir / 'report.html'
        self.report_service.create_html_report(
            data={
                'project': config.get('project', 'M3C2 Analysis'),
                'statistics': statistics,
                'plots': plots,
                'parameters': config
            },
            output_path=report_path,
            include_plots=True
        )
        results['report'] = report_path
        
        logger.info("Processing complete")
        return results
    
    def _flatten_statistics(self, statistics: Dict) -> pd.DataFrame:
        """Flacht verschachtelte Statistiken ab"""
        import pandas as pd
        
        rows = []
        for strategy_name, strategy_stats in statistics.items():
            if isinstance(strategy_stats, dict):
                for key, value in strategy_stats.items():
                    rows.append({
                        'Strategy': strategy_name,
                        'Metric': key,
                        'Value': value
                    })
        
        return pd.DataFrame(rows)
    
    def create_comparison_analysis(
        self,
        datasets: Dict[str, np.ndarray],
        output_dir: Path
    ) -> Dict[str, Any]:
        """Erstellt Vergleichsanalyse für mehrere Datensätze"""
        results = {}
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics comparison
        stats_df = self.statistics_service.calculate_batch_statistics(
            datasets,
            strategies=['basic', 'robust']
        )
        results['statistics'] = stats_df
        
        # Comparison plots
        plots = self.plot_service.create_comparison_suite(
            datasets,
            output_dir / 'comparison_plots'
        )
        results['plots'] = plots
        
        # Export
        excel_path = output_dir / 'comparison.xlsx'
        self.export_service.export_to_excel(
            {'Comparison': stats_df},
            excel_path
        )
        results['excel'] = excel_path
        
        return results