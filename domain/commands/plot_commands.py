# New_Architecture/domain/commands/plot_commands.py
"""Commands für Plot-Generierung"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
from domain.commands.base import Command, PipelineContext
from domain.entities import CloudPair

logger = logging.getLogger(__name__)


class GeneratePlotsCommand(Command):
    """Generiert verschiedene Plot-Typen für Distanzen"""
    
    def __init__(self, plot_service, plot_config: Optional[Dict] = None):
        super().__init__("GeneratePlots")
        self.plot_service = plot_service
        self.plot_config = plot_config or {}
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Generiert Plots basierend auf Konfiguration"""
        self.log_execution(context)
        
        config = context.get('config')
        distances = context.get('distances', {})
        
        if not distances:
            logger.warning("No distances found for plotting")
            return context
        
        output_dir = Path(config.output_dir) / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_results = {}
        
        # Histogram
        if self.plot_config.get('histogram', True):
            hist_path = self._generate_histogram(
                distances, output_dir, config.cloud_pair.tag
            )
            plot_results['histogram'] = hist_path
        
        # Gaussian fit
        if self.plot_config.get('gaussian', False):
            gauss_path = self._generate_gaussian_fit(
                distances, output_dir, config.cloud_pair.tag
            )
            plot_results['gaussian'] = gauss_path
        
        # Box plot
        if self.plot_config.get('boxplot', False):
            box_path = self._generate_boxplot(
                distances, output_dir, config.cloud_pair.tag
            )
            plot_results['boxplot'] = box_path
        
        # Q-Q plot
        if self.plot_config.get('qq_plot', False):
            qq_path = self._generate_qq_plot(
                distances, output_dir, config.cloud_pair.tag
            )
            plot_results['qq_plot'] = qq_path
        
        context.set('plot_results', plot_results)
        logger.info(f"Generated {len(plot_results)} plots")
        
        return context
    
    def _generate_histogram(self, distances: Dict, output_dir: Path, tag: str) -> Path:
        """Generiert Histogram"""
        output_path = output_dir / f"{tag}_histogram.png"
        
        # Separate WITH and INLIER data
        data_with = distances.get('with_outliers', np.array([]))
        data_inlier = distances.get('inliers', np.array([]))
        
        self.plot_service.create_histogram(
            data_with=data_with,
            data_inlier=data_inlier,
            output_path=output_path,
            title=f"Distance Distribution - {tag}",
            bins=self.plot_config.get('bins', 50)
        )
        
        return output_path
    
    def _generate_gaussian_fit(self, distances: Dict, output_dir: Path, tag: str) -> Path:
        """Generiert Gaussian Fit Plot"""
        output_path = output_dir / f"{tag}_gaussian.png"
        
        data = distances.get('inliers', distances.get('with_outliers', np.array([])))
        
        self.plot_service.create_gaussian_fit(
            data=data,
            output_path=output_path,
            title=f"Gaussian Fit - {tag}"
        )
        
        return output_path
    
    def _generate_boxplot(self, distances: Dict, output_dir: Path, tag: str) -> Path:
        """Generiert Box Plot"""
        output_path = output_dir / f"{tag}_boxplot.png"
        
        data_dict = {
            'With Outliers': distances.get('with_outliers', np.array([])),
            'Inliers Only': distances.get('inliers', np.array([]))
        }
        
        self.plot_service.create_boxplot(
            data_dict=data_dict,
            output_path=output_path,
            title=f"Box Plot - {tag}"
        )
        
        return output_path
    
    def _generate_qq_plot(self, distances: Dict, output_dir: Path, tag: str) -> Path:
        """Generiert Q-Q Plot"""
        output_path = output_dir / f"{tag}_qq.png"
        
        data = distances.get('inliers', distances.get('with_outliers', np.array([])))
        
        self.plot_service.create_qq_plot(
            data=data,
            output_path=output_path,
            title=f"Q-Q Plot - {tag}"
        )
        
        return output_path
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Prüft ob Plots generiert werden können"""
        return context.has('distances') or context.has('m3c2_distances')


class GenerateComparisonPlotsCommand(Command):
    """Generiert Vergleichs-Plots zwischen verschiedenen Distanz-Sets"""
    
    def __init__(self, plot_service, comparison_config: Optional[Dict] = None):
        super().__init__("GenerateComparisonPlots")
        self.plot_service = plot_service
        self.comparison_config = comparison_config or {}
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Generiert Vergleichs-Plots"""
        self.log_execution(context)
        
        # Hole alle Distanz-Sets aus dem Kontext
        all_distances = context.get('all_distances', {})
        if not all_distances:
            logger.warning("No distance sets found for comparison")
            return context
        
        config = context.get('config')
        output_dir = Path(config.output_dir) / "comparison_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        comparison_results = {}
        
        # Bland-Altman Plot
        if self.comparison_config.get('bland_altman', True):
            ba_paths = self._generate_bland_altman(all_distances, output_dir)
            comparison_results['bland_altman'] = ba_paths
        
        # Passing-Bablok Regression
        if self.comparison_config.get('passing_bablok', False):
            pb_paths = self._generate_passing_bablok(all_distances, output_dir)
            comparison_results['passing_bablok'] = pb_paths
        
        # Linear Regression
        if self.comparison_config.get('linear_regression', False):
            lr_paths = self._generate_linear_regression(all_distances, output_dir)
            comparison_results['linear_regression'] = lr_paths
        
        context.set('comparison_plots', comparison_results)
        logger.info(f"Generated {len(comparison_results)} comparison plot types")
        
        return context
    
    def _generate_bland_altman(self, all_distances: Dict, output_dir: Path) -> List[Path]:
        """Generiert Bland-Altman Plots für alle Paare"""
        paths = []
        distance_keys = list(all_distances.keys())
        
        for i in range(len(distance_keys)):
            for j in range(i + 1, len(distance_keys)):
                key1, key2 = distance_keys[i], distance_keys[j]
                output_path = output_dir / f"bland_altman_{key1}_vs_{key2}.png"
                
                self.plot_service.create_bland_altman(
                    data1=all_distances[key1],
                    data2=all_distances[key2],
                    output_path=output_path,
                    title=f"Bland-Altman: {key1} vs {key2}"
                )
                paths.append(output_path)
        
        return paths
    
    def _generate_passing_bablok(self, all_distances: Dict, output_dir: Path) -> List[Path]:
        """Generiert Passing-Bablok Regression Plots"""
        paths = []
        distance_keys = list(all_distances.keys())
        
        for i in range(len(distance_keys)):
            for j in range(i + 1, len(distance_keys)):
                key1, key2 = distance_keys[i], distance_keys[j]
                output_path = output_dir / f"passing_bablok_{key1}_vs_{key2}.png"
                
                self.plot_service.create_passing_bablok(
                    data1=all_distances[key1],
                    data2=all_distances[key2],
                    output_path=output_path,
                    title=f"Passing-Bablok: {key1} vs {key2}"
                )
                paths.append(output_path)
        
        return paths
    
    def _generate_linear_regression(self, all_distances: Dict, output_dir: Path) -> List[Path]:
        """Generiert Linear Regression Plots"""
        paths = []
        distance_keys = list(all_distances.keys())
        
        for i in range(len(distance_keys)):
            for j in range(i + 1, len(distance_keys)):
                key1, key2 = distance_keys[i], distance_keys[j]
                output_path = output_dir / f"linear_reg_{key1}_vs_{key2}.png"
                
                self.plot_service.create_linear_regression(
                    data1=all_distances[key1],
                    data2=all_distances[key2],
                    output_path=output_path,
                    title=f"Linear Regression: {key1} vs {key2}"
                )
                paths.append(output_path)
        
        return paths
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Prüft ob Vergleichs-Plots generiert werden können"""
        all_distances = context.get('all_distances', {})
        return len(all_distances) >= 2


class GenerateBatchPlotsCommand(Command):
    """Generiert Plots für Batch-Verarbeitung über mehrere Indizes"""
    
    def __init__(self, plot_service, batch_config: Optional[Dict] = None):
        super().__init__("GenerateBatchPlots")
        self.plot_service = plot_service
        self.batch_config = batch_config or {}
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Generiert Batch-Plots für alle Indizes"""
        self.log_execution(context)
        
        # Hole gruppierte Daten nach Index
        grouped_data = context.get('grouped_by_index', {})
        if not grouped_data:
            logger.warning("No grouped data found for batch plotting")
            return context
        
        config = context.get('config')
        output_dir = Path(config.output_dir) / "batch_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        batch_results = {}
        
        for index, index_data in grouped_data.items():
            index_dir = output_dir / f"part_{index}"
            index_dir.mkdir(exist_ok=True)
            
            # Generiere Plots für jeden Index
            index_plots = self._generate_index_plots(
                index_data, index_dir, index
            )
            batch_results[f"part_{index}"] = index_plots
        
        # Generiere Summary Plots
        if self.batch_config.get('generate_summary', True):
            summary_plots = self._generate_summary_plots(
                grouped_data, output_dir
            )
            batch_results['summary'] = summary_plots
        
        context.set('batch_plot_results', batch_results)
        logger.info(f"Generated batch plots for {len(grouped_data)} indices")
        
        return context
    
    def _generate_index_plots(self, index_data: Dict, output_dir: Path, index: int) -> Dict:
        """Generiert alle Plots für einen Index"""
        plots = {}
        
        # Histogram für WITH und INLIER
        if 'WITH' in index_data:
            hist_path = output_dir / f"histogram_with_outliers.png"
            self.plot_service.create_overlay_histogram(
                data_dict=index_data['WITH'],
                output_path=hist_path,
                title=f"Part {index} - With Outliers",
                bins=self.batch_config.get('bins', 50)
            )
            plots['histogram_with'] = hist_path
        
        if 'INLIER' in index_data:
            hist_path = output_dir / f"histogram_inliers.png"
            self.plot_service.create_overlay_histogram(
                data_dict=index_data['INLIER'],
                output_path=hist_path,
                title=f"Part {index} - Inliers Only",
                bins=self.batch_config.get('bins', 50)
            )
            plots['histogram_inlier'] = hist_path
        
        # Grouped Bar Chart
        if self.batch_config.get('grouped_bar', True):
            bar_path = output_dir / f"grouped_bar.png"
            self.plot_service.create_grouped_bar_chart(
                data_with=index_data.get('WITH', {}),
                data_inlier=index_data.get('INLIER', {}),
                output_path=bar_path,
                title=f"Part {index} - Mean & Std Comparison"
            )
            plots['grouped_bar'] = bar_path
        
        return plots
    
    def _generate_summary_plots(self, grouped_data: Dict, output_dir: Path) -> Dict:
        """Generiert Summary Plots über alle Indizes"""
        summary = {}
        
        # Violin Plot über alle Indizes
        if self.batch_config.get('violin_plot', True):
            violin_path = output_dir / "summary_violin.png"
            self.plot_service.create_violin_plot(
                grouped_data=grouped_data,
                output_path=violin_path,
                title="All Parts - Distribution Comparison"
            )
            summary['violin'] = violin_path
        
        # Trend Plot über Indizes
        if self.batch_config.get('trend_plot', True):
            trend_path = output_dir / "summary_trend.png"
            self.plot_service.create_trend_plot(
                grouped_data=grouped_data,
                output_path=trend_path,
                title="Trend Analysis Across Parts"
            )
            summary['trend'] = trend_path
        
        return summary
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Prüft ob Batch-Plots generiert werden können"""
        return context.has('grouped_by_index') or context.has('batch_results')


class ExportPlotsToPDFCommand(Command):
    """Exportiert alle generierten Plots in eine PDF-Datei"""
    
    def __init__(self, plot_service):
        super().__init__("ExportPlotsToPDF")
        self.plot_service = plot_service
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Exportiert Plots zu PDF"""
        self.log_execution(context)
        
        config = context.get('config')
        output_dir = Path(config.output_dir)
        
        # Sammle alle Plot-Pfade
        all_plots = []
        
        # Einzelne Plots
        if context.has('plot_results'):
            plots = context.get('plot_results')
            all_plots.extend([p for p in plots.values() if isinstance(p, Path)])
        
        # Batch Plots
        if context.has('batch_plot_results'):
            batch_plots = context.get('batch_plot_results')
            for index_plots in batch_plots.values():
                if isinstance(index_plots, dict):
                    all_plots.extend([p for p in index_plots.values() if isinstance(p, Path)])
        
        if not all_plots:
            logger.warning("No plots found for PDF export")
            return context
        
        # Generiere PDFs
        pdf_paths = []
        
        # PDF mit Outliers
        pdf_with = output_dir / f"{config.cloud_pair.tag}_plots_with_outliers.pdf"
        self.plot_service.build_pdf(
            plot_paths=[p for p in all_plots if 'with' in str(p).lower() or 'WITH' in str(p)],
            output_path=pdf_with,
            title=f"{config.cloud_pair.tag} - Including Outliers"
        )
        pdf_paths.append(pdf_with)
        
        # PDF ohne Outliers
        pdf_without = output_dir / f"{config.cloud_pair.tag}_plots_inliers.pdf"
        self.plot_service.build_pdf(
            plot_paths=[p for p in all_plots if 'inlier' in str(p).lower() or 'INLIER' in str(p)],
            output_path=pdf_without,
            title=f"{config.cloud_pair.tag} - Inliers Only"
        )
        pdf_paths.append(pdf_without)
        
        context.set('pdf_exports', pdf_paths)
        logger.info(f"Exported plots to {len(pdf_paths)} PDF files")
        
        return context
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Prüft ob PDFs generiert werden können"""
        return context.has('plot_results') or context.has('batch_plot_results')