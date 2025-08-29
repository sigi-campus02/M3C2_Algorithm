# New_Architecture/domain/strategies/plot_strategies.py
"""Strategy Pattern für verschiedene Plot-Typen"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, weibull_min, gamma, lognorm
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class PlotStrategy(ABC):
    """Abstrakte Basis für Plot-Strategien"""
    
    @abstractmethod
    def create_plot(self, data: Any, output_path: Path, **kwargs) -> None:
        """Erstellt einen Plot"""
        pass
    
    @abstractmethod
    def get_plot_type(self) -> str:
        """Gibt den Plot-Typ zurück"""
        pass


class HistogramStrategy(PlotStrategy):
    """Strategy für Histogram-Plots"""
    
    def __init__(self, style_config: Optional[Dict] = None):
        self.style_config = style_config or {}
        self.default_bins = 50
    
    def create_plot(self, data: Any, output_path: Path, **kwargs) -> None:
        """Erstellt Histogram"""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        bins = kwargs.get('bins', self.default_bins)
        title = kwargs.get('title', 'Histogram')
        xlabel = kwargs.get('xlabel', 'Value')
        ylabel = kwargs.get('ylabel', 'Frequency')
        
        if isinstance(data, dict):
            # Multiple datasets
            for label, values in data.items():
                if len(values) > 0:
                    ax.hist(values, bins=bins, alpha=0.5, label=label)
            ax.legend()
        else:
            # Single dataset
            ax.hist(data, bins=bins, alpha=0.7)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Created histogram: {output_path}")
    
    def get_plot_type(self) -> str:
        return "histogram"


class GaussianFitStrategy(PlotStrategy):
    """Strategy für Gaussian Fit Plots"""
    
    def create_plot(self, data: Any, output_path: Path, **kwargs) -> None:
        """Erstellt Gaussian Fit Plot"""
        if len(data) == 0:
            logger.warning("No data for Gaussian fit")
            return
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        # Fit Gaussian
        mu, sigma = norm.fit(data)
        
        # Plot histogram
        n, bins, patches = ax.hist(data, bins=50, density=True, alpha=0.6,
                                  color=kwargs.get('color', 'blue'))
        
        # Plot fitted curve
        x = np.linspace(data.min(), data.max(), 100)
        fitted_curve = norm.pdf(x, mu, sigma)
        ax.plot(x, fitted_curve, 'r-', linewidth=2,
               label=f'Gaussian: μ={mu:.4f}, σ={sigma:.4f}')
        
        ax.set_xlabel(kwargs.get('xlabel', 'Value'))
        ax.set_ylabel(kwargs.get('ylabel', 'Probability Density'))
        ax.set_title(kwargs.get('title', 'Gaussian Fit'))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_plot_type(self) -> str:
        return "gaussian_fit"


class WeibullFitStrategy(PlotStrategy):
    """Strategy für Weibull Fit Plots"""
    
    def create_plot(self, data: Any, output_path: Path, **kwargs) -> None:
        """Erstellt Weibull Fit Plot"""
        if len(data) == 0:
            logger.warning("No data for Weibull fit")
            return
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        # Ensure positive values
        data_positive = np.abs(data)
        
        # Fit Weibull
        params = weibull_min.fit(data_positive, floc=0)
        shape, loc, scale = params
        
        # Plot histogram
        n, bins, patches = ax.hist(data_positive, bins=50, density=True, alpha=0.6,
                                  color=kwargs.get('color', 'green'))
        
        # Plot fitted curve
        x = np.linspace(data_positive.min(), data_positive.max(), 100)
        fitted_curve = weibull_min.pdf(x, shape, loc, scale)
        ax.plot(x, fitted_curve, 'r-', linewidth=2,
               label=f'Weibull: shape={shape:.3f}, scale={scale:.3f}')
        
        ax.set_xlabel(kwargs.get('xlabel', 'Value'))
        ax.set_ylabel(kwargs.get('ylabel', 'Probability Density'))
        ax.set_title(kwargs.get('title', 'Weibull Fit'))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_plot_type(self) -> str:
        return "weibull_fit"


class BoxPlotStrategy(PlotStrategy):
    """Strategy für Box Plots"""
    
    def create_plot(self, data: Any, output_path: Path, **kwargs) -> None:
        """Erstellt Box Plot"""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        if isinstance(data, dict):
            # Multiple datasets
            plot_data = []
            labels = []
            for label, values in data.items():
                if len(values) > 0:
                    plot_data.append(values)
                    labels.append(label)
            
            if plot_data:
                bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
                
                # Color boxes
                colors = kwargs.get('colors', plt.cm.Set3(np.linspace(0, 1, len(plot_data))))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
        else:
            # Single dataset
            ax.boxplot([data], patch_artist=True)
        
        ax.set_ylabel(kwargs.get('ylabel', 'Value'))
        ax.set_title(kwargs.get('title', 'Box Plot'))
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_plot_type(self) -> str:
        return "boxplot"


class QQPlotStrategy(PlotStrategy):
    """Strategy für Q-Q Plots"""
    
    def create_plot(self, data: Any, output_path: Path, **kwargs) -> None:
        """Erstellt Q-Q Plot"""
        if len(data) == 0:
            logger.warning("No data for Q-Q plot")
            return
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 8)))
        
        distribution = kwargs.get('distribution', 'norm')
        stats.probplot(data, dist=distribution, plot=ax)
        
        ax.set_title(kwargs.get('title', f'Q-Q Plot ({distribution})'))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_plot_type(self) -> str:
        return "qq_plot"


class ViolinPlotStrategy(PlotStrategy):
    """Strategy für Violin Plots"""
    
    def create_plot(self, data: Any, output_path: Path, **kwargs) -> None:
        """Erstellt Violin Plot"""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 6)))
        
        if isinstance(data, dict):
            # Convert dict to list format
            plot_data = []
            labels = []
            for label, values in data.items():
                if len(values) > 0:
                    plot_data.append(values)
                    labels.append(label)
            
            if plot_data:
                parts = ax.violinplot(plot_data, positions=range(len(plot_data)),
                                     showmeans=True, showmedians=True, showextrema=True)
                
                # Customize colors
                for pc in parts['bodies']:
                    pc.set_facecolor(kwargs.get('color', '#8dd3c7'))
                    pc.set_alpha(0.7)
                
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
        else:
            parts = ax.violinplot([data], positions=[0],
                                 showmeans=True, showmedians=True)
        
        ax.set_ylabel(kwargs.get('ylabel', 'Value'))
        ax.set_title(kwargs.get('title', 'Violin Plot'))
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_plot_type(self) -> str:
        return "violin_plot"


class HeatmapStrategy(PlotStrategy):
    """Strategy für Heatmap Plots"""
    
    def create_plot(self, data: Any, output_path: Path, **kwargs) -> None:
        """Erstellt Heatmap"""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))
        
        # Ensure data is 2D
        if len(data.shape) == 1:
            # Convert to correlation matrix if 1D
            data = np.corrcoef(data.reshape(-1, 1))
        
        im = ax.imshow(data, cmap=kwargs.get('cmap', 'coolwarm'), aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        ax.set_title(kwargs.get('title', 'Heatmap'))
        
        # Add labels if provided
        if 'xlabels' in kwargs:
            ax.set_xticks(range(len(kwargs['xlabels'])))
            ax.set_xticklabels(kwargs['xlabels'], rotation=45, ha='right')
        
        if 'ylabels' in kwargs:
            ax.set_yticks(range(len(kwargs['ylabels'])))
            ax.set_yticklabels(kwargs['ylabels'])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_plot_type(self) -> str:
        return "heatmap"


class ScatterPlotStrategy(PlotStrategy):
    """Strategy für Scatter Plots"""
    
    def create_plot(self, data: Any, output_path: Path, **kwargs) -> None:
        """Erstellt Scatter Plot"""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        if isinstance(data, tuple) and len(data) == 2:
            x, y = data
        elif isinstance(data, np.ndarray) and data.shape[1] == 2:
            x, y = data[:, 0], data[:, 1]
        else:
            logger.error("Invalid data format for scatter plot")
            return
        
        # Create scatter plot
        scatter = ax.scatter(x, y, 
                           c=kwargs.get('colors', 'blue'),
                           s=kwargs.get('size', 20),
                           alpha=kwargs.get('alpha', 0.6))
        
        # Add regression line if requested
        if kwargs.get('regression', False):
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", alpha=0.8, 
                   label=f'y={z[0]:.3f}x+{z[1]:.3f}')
            ax.legend()
        
        ax.set_xlabel(kwargs.get('xlabel', 'X'))
        ax.set_ylabel(kwargs.get('ylabel', 'Y'))
        ax.set_title(kwargs.get('title', 'Scatter Plot'))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_plot_type(self) -> str:
        return "scatter_plot"


class BlandAltmanStrategy(PlotStrategy):
    """Strategy für Bland-Altman Plots"""
    
    def create_plot(self, data: Any, output_path: Path, **kwargs) -> None:
        """Erstellt Bland-Altman Plot"""
        if isinstance(data, tuple) and len(data) == 2:
            data1, data2 = data
        else:
            logger.error("Bland-Altman requires two datasets")
            return
        
        # Ensure same length
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        mean = (data1 + data2) / 2
        diff = data1 - data2
        
        md = np.mean(diff)
        sd = np.std(diff)
        
        # Scatter plot
        ax.scatter(mean, diff, alpha=0.5, s=20)
        
        # Mean line
        ax.axhline(md, color='red', linestyle='-', linewidth=2,
                  label=f'Mean: {md:.4f}')
        
        # Limits of agreement
        ax.axhline(md + 1.96*sd, color='red', linestyle='--', linewidth=1,
                  label=f'±1.96 SD: ±{1.96*sd:.4f}')
        ax.axhline(md - 1.96*sd, color='red', linestyle='--', linewidth=1)
        
        # Add confidence bands if requested
        if kwargs.get('confidence_bands', False):
            ax.fill_between(mean, md - 1.96*sd, md + 1.96*sd, 
                          alpha=0.1, color='red')
        
        ax.set_xlabel(kwargs.get('xlabel', 'Mean of two measurements'))
        ax.set_ylabel(kwargs.get('ylabel', 'Difference between measurements'))
        ax.set_title(kwargs.get('title', 'Bland-Altman Plot'))
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_plot_type(self) -> str:
        return "bland_altman"


# Factory für Plot-Strategien
class PlotStrategyFactory:
    """Factory für Plot-Strategien"""
    
    _strategies = {
        'histogram': HistogramStrategy,
        'gaussian': GaussianFitStrategy,
        'weibull': WeibullFitStrategy,
        'boxplot': BoxPlotStrategy,
        'qq': QQPlotStrategy,
        'violin': ViolinPlotStrategy,
        'heatmap': HeatmapStrategy,
        'scatter': ScatterPlotStrategy,
        'bland_altman': BlandAltmanStrategy
    }
    
    @classmethod
    def create_strategy(cls, plot_type: str, **kwargs) -> PlotStrategy:
        """Erstellt eine Plot-Strategy"""
        strategy_class = cls._strategies.get(plot_type.lower())
        if not strategy_class:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        return strategy_class(**kwargs)
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type) -> None:
        """Registriert eine neue Strategy"""
        if not issubclass(strategy_class, PlotStrategy):
            raise ValueError("Strategy must inherit from PlotStrategy")
        
        cls._strategies[name.lower()] = strategy_class
        logger.info(f"Registered plot strategy: {name}")
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Gibt verfügbare Strategien zurück"""
        return list(cls._strategies.keys())


# Theme Manager für konsistentes Styling
class PlotThemeManager:
    """Manager für Plot-Themes"""
    
    def __init__(self, theme: str = 'default'):
        self.themes = {
            'default': {
                'style': 'seaborn-v0_8-darkgrid',
                'palette': 'husl',
                'figsize': (10, 6),
                'dpi': 300,
                'colors': {
                    'primary': '#1f77b4',
                    'secondary': '#ff7f0e',
                    'success': '#2ca02c',
                    'danger': '#d62728',
                    'warning': '#ff9800',
                    'info': '#17a2b8'
                }
            },
            'dark': {
                'style': 'dark_background',
                'palette': 'viridis',
                'figsize': (10, 6),
                'dpi': 300,
                'colors': {
                    'primary': '#00bcd4',
                    'secondary': '#ffc107',
                    'success': '#4caf50',
                    'danger': '#f44336',
                    'warning': '#ff9800',
                    'info': '#2196f3'
                }
            },
            'minimal': {
                'style': 'seaborn-v0_8-whitegrid',
                'palette': 'muted',
                'figsize': (10, 6),
                'dpi': 300,
                'colors': {
                    'primary': '#4a90e2',
                    'secondary': '#7b68ee',
                    'success': '#5cb85c',
                    'danger': '#d9534f',
                    'warning': '#f0ad4e',
                    'info': '#5bc0de'
                }
            },
            'publication': {
                'style': 'seaborn-v0_8-ticks',
                'palette': 'colorblind',
                'figsize': (8, 6),
                'dpi': 600,
                'colors': {
                    'primary': '#0173b2',
                    'secondary': '#de8f05',
                    'success': '#029e73',
                    'danger': '#cc78bc',
                    'warning': '#ece133',
                    'info': '#56b4e9'
                }
            }
        }
        self.set_theme(theme)
    
    def set_theme(self, theme: str) -> None:
        """Setzt das aktuelle Theme"""
        if theme not in self.themes:
            raise ValueError(f"Unknown theme: {theme}")
        
        self.current_theme = self.themes[theme]
        
        # Apply matplotlib style
        try:
            plt.style.use(self.current_theme['style'])
        except:
            plt.style.use('default')
        
        # Set seaborn palette
        sns.set_palette(self.current_theme['palette'])
    
    def get_color(self, color_type: str) -> str:
        """Gibt eine Theme-Farbe zurück"""
        return self.current_theme['colors'].get(color_type, '#000000')
    
    def get_figsize(self) -> Tuple[int, int]:
        """Gibt die Standard-Figurengröße zurück"""
        return self.current_theme['figsize']
    
    def get_dpi(self) -> int:
        """Gibt die Standard-DPI zurück"""
        return self.current_theme['dpi']
    
    def apply_to_axes(self, ax) -> None:
        """Wendet Theme auf Axes an"""
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def create_custom_theme(self, name: str, config: Dict) -> None:
        """Erstellt ein Custom Theme"""
        self.themes[name] = config
        logger.info(f"Created custom theme: {name}")