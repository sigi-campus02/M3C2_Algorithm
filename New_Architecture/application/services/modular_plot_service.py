# New_Architecture/application/services/modular_plot_service.py
"""Modularisierter Plot Service mit Strategy Pattern"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
from domain.strategies.plot_strategies import (
    PlotStrategy, PlotStrategyFactory, PlotThemeManager
)

logger = logging.getLogger(__name__)


class ModularPlotService:
    """Modularisierter Service für Plot-Generierung"""
    
    def __init__(
        self, 
        repository,
        theme: str = 'default',
        config: Optional[Dict] = None
    ):
        self.repository = repository
        self.config = config or {}
        self.theme_manager = PlotThemeManager(theme)
        self.strategy_factory = PlotStrategyFactory()
        self._cache = {}
        
        logger.info(f"Initialized ModularPlotService with theme: {theme}")
    
    def create_plot(
        self,
        plot_type: str,
        data: Any,
        output_path: Path,
        **kwargs
    ) -> Path:
        """Erstellt einen Plot mit der entsprechenden Strategy"""
        try:
            # Get strategy
            strategy = self.strategy_factory.create_strategy(plot_type)
            
            # Apply theme settings
            kwargs.setdefault('figsize', self.theme_manager.get_figsize())
            kwargs.setdefault('dpi', self.theme_manager.get_dpi())
            
            # Create plot
            strategy.create_plot(data, output_path, **kwargs)
            
            # Cache result
            self._cache[str(output_path)] = {
                'type': plot_type,
                'timestamp': Path(output_path).stat().st_mtime
            }
            
            logger.info(f"Created {plot_type} plot: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create {plot_type} plot: {e}")
            raise
    
    def create_multiple_plots(
        self,
        plot_configs: List[Dict],
        output_dir: Path
    ) -> Dict[str, Path]:
        """Erstellt mehrere Plots"""
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        
        for config in plot_configs:
            plot_type = config.pop('type')
            filename = config.pop('filename', f"{plot_type}.png")
            output_path = output_dir / filename
            
            try:
                path = self.create_plot(
                    plot_type=plot_type,
                    output_path=output_path,
                    **config
                )
                results[plot_type] = path
            except Exception as e:
                logger.error(f"Failed to create {plot_type}: {e}")
                results[plot_type] = None
        
        return results
    
    def create_comparison_suite(
        self,
        data_sets: Dict[str, np.ndarray],
        output_dir: Path,
        plot_types: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """Erstellt eine Suite von Vergleichs-Plots"""
        if plot_types is None:
            plot_types = ['histogram', 'boxplot', 'violin', 'qq']
        
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        
        for plot_type in plot_types:
            output_path = output_dir / f"comparison_{plot_type}.png"
            
            try:
                if plot_type in ['histogram', 'boxplot', 'violin']:
                    # These support multiple datasets
                    path = self.create_plot(
                        plot_type=plot_type,
                        data=data_sets,
                        output_path=output_path,
                        title=f"Comparison - {plot_type.title()}"
                    )
                    results[plot_type] = path
                elif plot_type == 'qq':
                    # QQ plot for each dataset
                    for name, data in data_sets.items():
                        qq_path = output_dir / f"qq_{name}.png"
                        path = self.create_plot(
                            plot_type='qq',
                            data=data,
                            output_path=qq_path,
                            title=f"Q-Q Plot - {name}"
                        )
                        results[f"qq_{name}"] = path
            except Exception as e:
                logger.error(f"Failed to create {plot_type}: {e}")
        
        return results
    
    def set_theme(self, theme: str) -> None:
        """Ändert das aktuelle Theme"""
        self.theme_manager.set_theme(theme)
        logger.info(f"Changed plot theme to: {theme}")
    
    def register_custom_strategy(
        self,
        name: str,
        strategy_class: type
    ) -> None:
        """Registriert eine Custom Plot Strategy"""
        self.strategy_factory.register_strategy(name, strategy_class)
    
    def get_available_plot_types(self) -> List[str]:
        """Gibt verfügbare Plot-Typen zurück"""
        return self.strategy_factory.get_available_strategies()
    
    def clear_cache(self) -> None:
        """Leert den Plot-Cache"""
        self._cache.clear()
        logger.info("Cleared plot cache")


# New_Architecture/application/services/modular_statistics_service.py
"""Modularisierter Statistics Service mit Strategy Pattern"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from domain.strategies.statistics_strategies import (
    StatisticsStrategy, StatisticsStrategyFactory, StatisticsAggregator
)

logger = logging.getLogger(__name__)


class ModularStatisticsService:
    """Modularisierter Service für Statistik-Berechnungen"""
    
    def __init__(
        self,
        repository,
        default_strategies: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ):
        self.repository = repository
        self.config = config or {}
        self.strategy_factory = StatisticsStrategyFactory()
        
        # Default strategies
        if default_strategies is None:
            default_strategies = ['basic', 'advanced', 'distance']
        
        self.default_strategies = [
            self.strategy_factory.create_strategy(s) 
            for s in default_strategies
        ]
        
        self.aggregator = StatisticsAggregator(self.default_strategies)
        self._cache = {}
        
        logger.info(f"Initialized ModularStatisticsService with {len(self.default_strategies)} strategies")
    
    def calculate_statistics(
        self,
        data: np.ndarray,
        strategies: Optional[List[str]] = None,
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Berechnet Statistiken mit angegebenen Strategien"""
        # Check cache
        if cache_key and cache_key in self._cache:
            logger.debug(f"Returning cached statistics for: {cache_key}")
            return self._cache[cache_key]
        
        # Use specified strategies or defaults
        if strategies:
            strategy_objects = [
                self.strategy_factory.create_strategy(s) 
                for s in strategies
            ]
            aggregator = StatisticsAggregator(strategy_objects)
        else:
            aggregator = self.aggregator
        
        # Calculate
        results = aggregator.calculate_all(data)
        
        # Cache if requested
        if cache_key:
            self._cache[cache_key] = results
        
        logger.info(f"Calculated statistics with {len(results)} strategy groups")
        return results
    
    def calculate_comparison_statistics(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        comparison_type: str = 'full'
    ) -> Dict[str, Any]:
        """Berechnet Vergleichs-Statistiken zwischen zwei Datensätzen"""
        results = {}
        
        # Individual statistics
        results['dataset1'] = self.calculate_statistics(data1, ['basic', 'robust'])
        results['dataset2'] = self.calculate_statistics(data2, ['basic', 'robust'])
        
        # Comparison statistics
        comparison_strategy = self.strategy_factory.create_strategy('comparison')
        results['comparison'] = comparison_strategy.calculate((data1, data2))
        
        # Additional tests if requested
        if comparison_type == 'full':
            # Normality tests for both
            normality_strategy = self.strategy_factory.create_strategy('normality')
            results['normality_data1'] = normality_strategy.calculate(data1)
            results['normality_data2'] = normality_strategy.calculate(data2)
        
        return results
    
    def calculate_batch_statistics(
        self,
        data_dict: Dict[str, np.ndarray],
        strategies: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Berechnet Statistiken für mehrere Datensätze"""
        rows = []
        
        for name, data in data_dict.items():
            stats = self.calculate_statistics(data, strategies)
            
            # Flatten nested structure
            row = {'name': name}
            for strategy_name, strategy_stats in stats.items():
                if isinstance(strategy_stats, dict):
                    for key, value in strategy_stats.items():
                        row[f"{strategy_name}_{key}"] = value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.set_index('name', inplace=True)
        
        return df
    
    def calculate_grouped_statistics(
        self,
        grouped_data: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, pd.DataFrame]:
        """Berechnet Statistiken für gruppierte Daten"""
        results = {}
        
        for group_name, group_data in grouped_data.items():
            df = self.calculate_batch_statistics(group_data)
            results[group_name] = df
            
            # Add summary row
            summary = df.mean()
            summary.name = 'MEAN'
            results[f"{group_name}_summary"] = summary
        
        return results
    
    def export_statistics(
        self,
        statistics: Dict[str, Any],
        output_path: Path,
        format: str = 'excel'
    ) -> None:
        """Exportiert Statistiken in verschiedene Formate"""
        if format == 'excel':
            self._export_to_excel(statistics, output_path)
        elif format == 'csv':
            self._export_to_csv(statistics, output_path)
        elif format == 'json':
            self._export_to_json(statistics, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_to_excel(self, statistics: Dict, output_path: Path) -> None:
        """Export zu Excel"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, data in statistics.items():
                if isinstance(data, pd.DataFrame):
                    data.to_excel(writer, sheet_name=sheet_name[:31])
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    
    def _export_to_csv(self, statistics: Dict, output_path: Path) -> None:
        """Export zu CSV"""
        # Flatten structure
        flattened = {}
        for key, value in statistics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flattened[f"{key}_{sub_key}"] = sub_value
            else:
                flattened[key] = value
        
        df = pd.DataFrame([flattened])
        df.to_csv(output_path, index=False)
    
    def _export_to_json(self, statistics: Dict, output_path: Path) -> None:
        """Export zu JSON"""
        import json
        
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        
        converted = convert(statistics)
        
        with open(output_path, 'w') as f:
            json.dump(converted, f, indent=2)
    
    def register_custom_strategy(
        self,
        name: str,
        strategy_class: type
    ) -> None:
        """Registriert eine Custom Statistics Strategy"""
        self.strategy_factory.register_strategy(name, strategy_class)
    
    def get_available_strategies(self) -> List[str]:
        """Gibt verfügbare Strategien zurück"""
        return self.strategy_factory.get_available_strategies()
    
    def clear_cache(self) -> None:
        """Leert den Statistics-Cache"""
        self._cache.clear()
        logger.info("Cleared statistics cache")


# New_Architecture/application/services/visualization_service.py
"""3D Visualization Service für Punktwolken"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class VisualizationService:
    """Service für 3D-Visualisierungen"""
    
    def __init__(
        self,
        repository,
        config: Optional[Dict] = None
    ):
        self.repository = repository
        self.config = config or {}
        self.default_colorscale = 'Viridis'
        
        logger.info("Initialized VisualizationService")
    
    def create_3d_scatter(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        output_path: Optional[Path] = None,
        title: str = "3D Point Cloud",
        **kwargs
    ) -> go.Figure:
        """Erstellt 3D Scatter Plot"""
        fig = go.Figure()
        
        # Prepare data
        if len(points.shape) != 2 or points.shape[1] != 3:
            raise ValueError("Points must be Nx3 array")
        
        # Create scatter plot
        scatter_kwargs = {
            'x': points[:, 0],
            'y': points[:, 1],
            'z': points[:, 2],
            'mode': 'markers',
            'marker': {
                'size': kwargs.get('marker_size', 2),
                'opacity': kwargs.get('opacity', 0.8)
            }
        }
        
        # Add colors if provided
        if colors is not None:
            scatter_kwargs['marker']['color'] = colors
            scatter_kwargs['marker']['colorscale'] = kwargs.get('colorscale', self.default_colorscale)
            scatter_kwargs['marker']['showscale'] = True
            scatter_kwargs['marker']['colorbar'] = {
                'title': kwargs.get('colorbar_title', 'Value')
            }
        
        fig.add_trace(go.Scatter3d(**scatter_kwargs))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=kwargs.get('xlabel', 'X'),
                yaxis_title=kwargs.get('ylabel', 'Y'),
                zaxis_title=kwargs.get('zlabel', 'Z'),
                aspectmode=kwargs.get('aspectmode', 'data')
            ),
            width=kwargs.get('width', 1000),
            height=kwargs.get('height', 800)
        )
        
        # Save if path provided
        if output_path:
            fig.write_html(str(output_path))
            logger.info(f"Saved 3D visualization to: {output_path}")
        
        return fig
    
    def create_distance_visualization(
        self,
        points: np.ndarray,
        distances: np.ndarray,
        output_path: Optional[Path] = None,
        title: str = "M3C2 Distances",
        threshold: Optional[float] = None
    ) -> go.Figure:
        """Visualisiert Punktwolke mit Distanzen"""
        fig = self.create_3d_scatter(
            points=points,
            colors=distances,
            title=title,
            colorbar_title="Distance [m]",
            colorscale='RdBu_r'
        )
        
        # Add threshold plane if specified
        if threshold is not None:
            self._add_threshold_plane(fig, points, threshold)
        
        if output_path:
            fig.write_html(str(output_path))
        
        return fig
    
    def create_comparison_visualization(
        self,
        cloud1: np.ndarray,
        cloud2: np.ndarray,
        output_path: Optional[Path] = None,
        title: str = "Cloud Comparison"
    ) -> go.Figure:
        """Visualisiert zwei Punktwolken zum Vergleich"""
        fig = go.Figure()
        
        # Add first cloud
        fig.add_trace(go.Scatter3d(
            x=cloud1[:, 0],
            y=cloud1[:, 1],
            z=cloud1[:, 2],
            mode='markers',
            name='Cloud 1',
            marker=dict(
                size=2,
                color='blue',
                opacity=0.5
            )
        ))
        
        # Add second cloud
        fig.add_trace(go.Scatter3d(
            x=cloud2[:, 0],
            y=cloud2[:, 1],
            z=cloud2[:, 2],
            mode='markers',
            name='Cloud 2',
            marker=dict(
                size=2,
                color='red',
                opacity=0.5
            )
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=1000,
            height=800,
            showlegend=True
        )
        
        if output_path:
            fig.write_html(str(output_path))
        
        return fig
    
    def create_cross_section(
        self,
        points: np.ndarray,
        distances: np.ndarray,
        axis: str = 'z',
        position: Optional[float] = None,
        thickness: float = 0.1,
        output_path: Optional[Path] = None
    ) -> go.Figure:
        """Erstellt Querschnitt-Visualisierung"""
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
        
        if position is None:
            position = np.median(points[:, axis_idx])
        
        # Select points in slice
        mask = np.abs(points[:, axis_idx] - position) < thickness / 2
        slice_points = points[mask]
        slice_distances = distances[mask]
        
        # Create 2D projection
        other_axes = [i for i in range(3) if i != axis_idx]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=slice_points[:, other_axes[0]],
            y=slice_points[:, other_axes[1]],
            mode='markers',
            marker=dict(
                size=5,
                color=slice_distances,
                colorscale='RdBu_r',
                showscale=True,
                colorbar=dict(title='Distance [m]')
            ),
            text=[f"Distance: {d:.4f}" for d in slice_distances],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        axis_labels = ['X', 'Y', 'Z']
        fig.update_layout(
            title=f"Cross Section at {axis.upper()}={position:.2f}",
            xaxis_title=axis_labels[other_axes[0]],
            yaxis_title=axis_labels[other_axes[1]],
            width=800,
            height=800,
            hovermode='closest'
        )
        
        if output_path:
            fig.write_html(str(output_path))
        
        return fig
    
    def create_dashboard(
        self,
        data: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Erstellt interaktives Dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('3D Point Cloud', 'Distance Distribution', 
                          'Statistics', 'Cross Section'),
            specs=[
                [{'type': 'scatter3d'}, {'type': 'histogram'}],
                [{'type': 'table'}, {'type': 'scatter'}]
            ]
        )
        
        # 3D scatter
        if 'points' in data and 'distances' in data:
            points = data['points']
            distances = data['distances']
            
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=distances,
                        colorscale='RdBu_r',
                        showscale=True
                    )
                ),
                row=1, col=1
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=distances, nbinsx=50),
                row=1, col=2
            )
        
        # Statistics table
        if 'statistics' in data:
            stats = data['statistics']
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value']),
                    cells=dict(
                        values=[
                            list(stats.keys()),
                            [f"{v:.4f}" if isinstance(v, float) else str(v) 
                             for v in stats.values()]
                        ]
                    )
                ),
                row=2, col=1
            )
        
        # Cross section
        if 'points' in data and 'distances' in data:
            # Simple 2D projection
            fig.add_trace(
                go.Scatter(
                    x=points[:, 0],
                    y=points[:, 1],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=distances,
                        colorscale='RdBu_r'
                    )
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='M3C2 Analysis Dashboard',
            height=900,
            showlegend=False
        )
        
        # Save
        fig.write_html(str(output_path))
        logger.info(f"Created dashboard: {output_path}")
    
    def _add_threshold_plane(
        self,
        fig: go.Figure,
        points: np.ndarray,
        threshold: float
    ) -> None:
        """Fügt Threshold-Ebene hinzu"""
        x_range = [points[:, 0].min(), points[:, 0].max()]
        y_range = [points[:, 1].min(), points[:, 1].max()]
        
        xx, yy = np.meshgrid(
            np.linspace(x_range[0], x_range[1], 10),
            np.linspace(y_range[0], y_range[1], 10)
        )
        
        zz = np.full_like(xx, threshold)
        
        fig.add_trace(go.Surface(
            x=xx,
            y=yy,
            z=zz,
            opacity=0.3,
            colorscale=[[0, 'red'], [1, 'red']],
            showscale=False,
            name=f'Threshold: {threshold}'
        ))
    
    def export_colored_pointcloud(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        output_path: Path,
        format: str = 'ply'
    ) -> None:
        """Exportiert farbige Punktwolke"""
        if format == 'ply':
            self._export_ply_with_colors(points, colors, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_ply_with_colors(
        self,
        points: np.ndarray,
        values: np.ndarray,
        output_path: Path
    ) -> None:
        """Exportiert PLY mit Farbwerten"""
        import plyfile
        
        # Normalize values to 0-255 range
        normalized = (values - values.min()) / (values.max() - values.min())
        colors = plt.cm.RdBu_r(normalized)[:, :3] * 255
        
        # Create structured array
        vertex = np.zeros(
            len(points),
            dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
            ]
        )
        
        vertex['x'] = points[:, 0]
        vertex['y'] = points[:, 1]
        vertex['z'] = points[:, 2]
        vertex['red'] = colors[:, 0]
        vertex['green'] = colors[:, 1]
        vertex['blue'] = colors[:, 2]
        
        # Write PLY
        el = plyfile.PlyElement.describe(vertex, 'vertex')
        plyfile.PlyData([el]).write(output_path)
        
        logger.info(f"Exported colored point cloud: {output_path}")