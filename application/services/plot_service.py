# New_Architecture/application/services/plot_service.py
"""Refactored Plot Service - Modular und testbar"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import norm, weibull_min
import seaborn as sns

logger = logging.getLogger(__name__)


class PlotService:
    """Service für Plot-Generierung"""
    
    def __init__(self, repository, config: Optional[Dict] = None):
        self.repository = repository
        self.config = config or {}
        self.default_colors = {
            'with_outliers': '#1f77b4',
            'inliers': '#2ca02c',
            'outliers': '#d62728'
        }
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def create_histogram(
        self,
        data_with: np.ndarray,
        data_inlier: np.ndarray,
        output_path: Path,
        title: str = "Distance Distribution",
        bins: int = 50
    ) -> None:
        """Erstellt Histogram mit Overlay"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histograms
        if len(data_with) > 0:
            ax.hist(data_with, bins=bins, alpha=0.5, 
                   label='With Outliers', color=self.default_colors['with_outliers'])
        
        if len(data_inlier) > 0:
            ax.hist(data_inlier, bins=bins, alpha=0.5,
                   label='Inliers Only', color=self.default_colors['inliers'])
        
        ax.set_xlabel('Distance [m]')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Created histogram: {output_path}")
    
    def create_gaussian_fit(
        self,
        data: np.ndarray,
        output_path: Path,
        title: str = "Gaussian Fit"
    ) -> None:
        """Erstellt Gaussian Fit Plot"""
        if len(data) == 0:
            logger.warning("No data for Gaussian fit")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Fit Gaussian
        mu, sigma = norm.fit(data)
        
        # Plot histogram
        n, bins, patches = ax.hist(data, bins=50, density=True, alpha=0.7,
                                  color=self.default_colors['inliers'])
        
        # Plot fitted curve
        x = np.linspace(data.min(), data.max(), 100)
        fitted_curve = norm.pdf(x, mu, sigma)
        ax.plot(x, fitted_curve, 'r-', linewidth=2,
               label=f'Gaussian: μ={mu:.4f}, σ={sigma:.4f}')
        
        ax.set_xlabel('Distance [m]')
        ax.set_ylabel('Probability Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Created Gaussian fit: {output_path}")
    
    def create_boxplot(
        self,
        data_dict: Dict[str, np.ndarray],
        output_path: Path,
        title: str = "Box Plot"
    ) -> None:
        """Erstellt Box Plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        plot_data = []
        labels = []
        colors = []
        
        for label, data in data_dict.items():
            if len(data) > 0:
                plot_data.append(data)
                labels.append(label)
                if 'outlier' in label.lower():
                    colors.append(self.default_colors['with_outliers'])
                else:
                    colors.append(self.default_colors['inliers'])
        
        if plot_data:
            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
            
            # Color boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_ylabel('Distance [m]')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Created boxplot: {output_path}")
    
    def create_qq_plot(
        self,
        data: np.ndarray,
        output_path: Path,
        title: str = "Q-Q Plot"
    ) -> None:
        """Erstellt Q-Q Plot"""
        if len(data) == 0:
            logger.warning("No data for Q-Q plot")
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Created Q-Q plot: {output_path}")
    
    def create_bland_altman(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        output_path: Path,
        title: str = "Bland-Altman Plot"
    ) -> None:
        """Erstellt Bland-Altman Plot"""
        if len(data1) == 0 or len(data2) == 0:
            logger.warning("Insufficient data for Bland-Altman plot")
            return
        
        # Ensure same length
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mean = (data1 + data2) / 2
        diff = data1 - data2
        
        md = np.mean(diff)
        sd = np.std(diff)
        
        ax.scatter(mean, diff, alpha=0.5)
        ax.axhline(md, color='red', linestyle='-', label=f'Mean: {md:.4f}')
        ax.axhline(md + 1.96*sd, color='red', linestyle='--', 
                  label=f'±1.96 SD: {1.96*sd:.4f}')
        ax.axhline(md - 1.96*sd, color='red', linestyle='--')
        
        ax.set_xlabel('Mean of two measurements')
        ax.set_ylabel('Difference between measurements')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Created Bland-Altman plot: {output_path}")
    
    def create_overlay_histogram(
        self,
        data_dict: Dict[str, np.ndarray],
        output_path: Path,
        title: str = "Overlay Histogram",
        bins: int = 50
    ) -> None:
        """Erstellt Overlay Histogram für mehrere Datensätze"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get common range
        all_data = np.concatenate([d for d in data_dict.values() if len(d) > 0])
        if len(all_data) == 0:
            logger.warning("No data for overlay histogram")
            return
        
        data_min, data_max = all_data.min(), all_data.max()
        bin_edges = np.linspace(data_min, data_max, bins + 1)
        
        # Plot each dataset
        for label, data in data_dict.items():
            if len(data) > 0:
                ax.hist(data, bins=bin_edges, alpha=0.5, label=label)
        
        ax.set_xlabel('Distance [m]')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Created overlay histogram: {output_path}")
    
    def create_grouped_bar_chart(
        self,
        data_with: Dict[str, np.ndarray],
        data_inlier: Dict[str, np.ndarray],
        output_path: Path,
        title: str = "Mean & Std Comparison"
    ) -> None:
        """Erstellt gruppiertes Balkendiagramm"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare data
        labels = list(set(data_with.keys()) | set(data_inlier.keys()))
        labels.sort()
        
        means_with = [np.mean(data_with.get(l, [])) if l in data_with and len(data_with[l]) > 0 else 0 for l in labels]
        means_inlier = [np.mean(data_inlier.get(l, [])) if l in data_inlier and len(data_inlier[l]) > 0 else 0 for l in labels]
        
        stds_with = [np.std(data_with.get(l, [])) if l in data_with and len(data_with[l]) > 0 else 0 for l in labels]
        stds_inlier = [np.std(data_inlier.get(l, [])) if l in data_inlier and len(data_inlier[l]) > 0 else 0 for l in labels]
        
        x = np.arange(len(labels))
        width = 0.35
        
        # Mean plot
        ax1.bar(x - width/2, means_with, width, label='With Outliers',
               color=self.default_colors['with_outliers'])
        ax1.bar(x + width/2, means_inlier, width, label='Inliers Only',
               color=self.default_colors['inliers'])
        
        ax1.set_xlabel('Group')
        ax1.set_ylabel('Mean Distance [m]')
        ax1.set_title('Mean Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Std plot
        ax2.bar(x - width/2, stds_with, width, label='With Outliers',
               color=self.default_colors['with_outliers'])
        ax2.bar(x + width/2, stds_inlier, width, label='Inliers Only',
               color=self.default_colors['inliers'])
        
        ax2.set_xlabel('Group')
        ax2.set_ylabel('Std Deviation [m]')
        ax2.set_title('Std Deviation Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Created grouped bar chart: {output_path}")
    
    def create_violin_plot(
        self,
        grouped_data: Dict[str, Dict],
        output_path: Path,
        title: str = "Distribution Comparison"
    ) -> None:
        """Erstellt Violin Plot"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        plot_data = []
        labels = []
        
        for group, data_dict in grouped_data.items():
            if 'INLIER' in data_dict:
                for case, distances in data_dict['INLIER'].items():
                    if len(distances) > 0:
                        plot_data.append(distances)
                        labels.append(f"{group}_{case}")
        
        if plot_data:
            parts = ax.violinplot(plot_data, positions=range(len(plot_data)),
                                 showmeans=True, showmedians=True)
            
            # Customize colors
            for pc in parts['bodies']:
                pc.set_facecolor(self.default_colors['inliers'])
                pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Distance [m]')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Created violin plot: {output_path}")
    
    def create_trend_plot(
        self,
        grouped_data: Dict[str, Dict],
        output_path: Path,
        title: str = "Trend Analysis"
    ) -> None:
        """Erstellt Trend-Analyse Plot"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        indices = []
        means_with = []
        means_inlier = []
        
        for group in sorted(grouped_data.keys()):
            # Extract index from group name
            try:
                idx = int(group.split('_')[1])
                indices.append(idx)
                
                # Calculate means
                if 'WITH' in grouped_data[group]:
                    all_with = np.concatenate(list(grouped_data[group]['WITH'].values()))
                    means_with.append(np.mean(all_with) if len(all_with) > 0 else 0)
                else:
                    means_with.append(0)
                
                if 'INLIER' in grouped_data[group]:
                    all_inlier = np.concatenate(list(grouped_data[group]['INLIER'].values()))
                    means_inlier.append(np.mean(all_inlier) if len(all_inlier) > 0 else 0)
                else:
                    means_inlier.append(0)
            except:
                continue
        
        if indices:
            ax.plot(indices, means_with, 'o-', label='With Outliers',
                   color=self.default_colors['with_outliers'], linewidth=2, markersize=8)
            ax.plot(indices, means_inlier, 's-', label='Inliers Only',
                   color=self.default_colors['inliers'], linewidth=2, markersize=8)
        
        ax.set_xlabel('Part Index')
        ax.set_ylabel('Mean Distance [m]')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Created trend plot: {output_path}")
    
    def build_pdf(
        self,
        plot_paths: List[Path],
        output_path: Path,
        title: str = "M3C2 Analysis Report"
    ) -> None:
        """Kombiniert mehrere Plots in eine PDF"""
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(output_path) as pdf:
            # Title page
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.5, title, ha='center', va='center', fontsize=24)
            pdf.savefig(fig)
            plt.close()
            
            # Add each plot
            for plot_path in plot_paths:
                if plot_path.exists():
                    try:
                        img = plt.imread(plot_path)
                        fig, ax = plt.subplots(figsize=(8.5, 11))
                        ax.imshow(img)
                        ax.axis('off')
                        ax.set_title(plot_path.stem.replace('_', ' ').title())
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close()
                    except Exception as e:
                        logger.warning(f"Could not add {plot_path} to PDF: {e}")
        
        logger.info(f"Created PDF report: {output_path}")


