# New_Architecture/application/services/enhanced_visualization_service.py
"""Enhanced Visualization Service mit CloudCompare-Kompatibilität"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import seaborn as sns

logger = logging.getLogger(__name__)

# Optional imports
try:
    from plyfile import PlyData, PlyElement
    PLYFILE_AVAILABLE = True
except ImportError:
    PLYFILE_AVAILABLE = False
    logger.warning("plyfile not available - PLY export limited")


class EnhancedVisualizationService:
    """
    Erweiterte Visualisierungs-Service mit:
    - CloudCompare-kompatiblen PLY Exports
    - Distance als Scalar Field
    - Outlier/Inlier Visualisierung
    - M3C2-spezifische Histogramme
    """
    
    def __init__(self):
        self.colormap = self._create_cloudcompare_colormap()
        
    def _create_cloudcompare_colormap(self) -> LinearSegmentedColormap:
        """Erstellt CloudCompare-ähnliche Farbskala: blau → grün → gelb → rot"""
        colors = [
            (0.0, "blue"),
            (0.33, "green"),
            (0.66, "yellow"),
            (1.0, "red")
        ]
        return LinearSegmentedColormap.from_list("CloudCompare", colors)
    
    def export_ply_with_scalar_field(
        self,
        points: np.ndarray,
        distances: np.ndarray,
        output_path: Path,
        scalar_name: str = "distance",
        nan_color: Tuple[int, int, int] = (255, 255, 255),
        percentile_range: Tuple[float, float] = (1.0, 99.0),
        binary: bool = True
    ) -> None:
        """
        Exportiert PLY mit Distance als Scalar Field für CloudCompare.
        
        Args:
            points: XYZ Koordinaten
            distances: M3C2 Distanzen
            output_path: Ausgabepfad
            scalar_name: Name des Scalar Fields
            nan_color: Farbe für NaN-Werte
            percentile_range: Clipping-Bereich für robuste Farbskala
            binary: Binary oder ASCII PLY
        """
        if not PLYFILE_AVAILABLE:
            raise RuntimeError("PLY export requires 'plyfile' package")
        
        n_points = len(points)
        if len(distances) != n_points:
            raise ValueError(f"Points ({n_points}) and distances ({len(distances)}) must have same length")
        
        # Berechne Farben basierend auf Distanzen
        colors = self._compute_distance_colors(
            distances, nan_color, percentile_range
        )
        
        # Erstelle strukturiertes Array mit Scalar Field
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            (scalar_name, 'f4')  # Scalar Field für CloudCompare
        ]
        
        vertex = np.zeros(n_points, dtype=dtype)
        vertex['x'] = points[:, 0]
        vertex['y'] = points[:, 1]
        vertex['z'] = points[:, 2]
        vertex['red'] = colors[:, 0]
        vertex['green'] = colors[:, 1]
        vertex['blue'] = colors[:, 2]
        vertex[scalar_name] = distances
        
        # Schreibe PLY
        el = PlyElement.describe(vertex, 'vertex')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        PlyData([el], text=not binary).write(str(output_path))
        
        logger.info(f"Exported PLY with scalar field '{scalar_name}': {output_path}")
    
    def _compute_distance_colors(
        self,
        distances: np.ndarray,
        nan_color: Tuple[int, int, int],
        percentile_range: Tuple[float, float]
    ) -> np.ndarray:
        """Berechnet Farben basierend auf Distanzen mit CloudCompare-Colormap"""
        n = len(distances)
        colors = np.zeros((n, 3), dtype=np.uint8)
        
        valid_mask = ~np.isnan(distances)
        if valid_mask.any():
            valid_distances = distances[valid_mask]
            
            # Robustes Clipping mit Perzentilen
            p_lo, p_hi = percentile_range
            vmin = float(np.percentile(valid_distances, p_lo))
            vmax = float(np.percentile(valid_distances, p_hi))
            
            if vmax <= vmin:
                vmax = vmin + 1e-12
            
            # Normalisiere auf [0, 1]
            normed = (np.clip(valid_distances, vmin, vmax) - vmin) / (vmax - vmin)
            
            # Wende Colormap an
            colored = (self.colormap(normed)[:, :3] * 255).astype(np.uint8)
            colors[valid_mask] = colored
        
        # NaN-Werte bekommen nan_color
        colors[~valid_mask] = np.array(nan_color, dtype=np.uint8)
        
        return colors
    
    def txt_to_ply_with_distance_color(
        self,
        txt_path: Path,
        output_path: Path,
        scalar_name: str = "distance",
        **kwargs
    ) -> None:
        """
        Konvertiert TXT (x y z distance) zu farbiger PLY mit Scalar Field.
        
        Args:
            txt_path: Eingabe TXT-Datei
            output_path: Ausgabe PLY-Datei
            scalar_name: Name des Scalar Fields
            **kwargs: Weitere Optionen für export_ply_with_scalar_field
        """
        # Lade Daten
        data = np.loadtxt(str(txt_path), skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if data.shape[1] < 4:
            raise ValueError(f"TXT file must have at least 4 columns (x,y,z,distance): {txt_path}")
        
        points = data[:, :3]
        distances = data[:, 3]
        
        # Exportiere als PLY
        self.export_ply_with_scalar_field(
            points, distances, output_path, scalar_name, **kwargs
        )
        
        logger.info(f"Converted {txt_path} to {output_path}")
    
    def generate_outlier_inlier_plys(
        self,
        points: np.ndarray,
        distances: np.ndarray,
        outlier_mask: np.ndarray,
        output_dir: Path,
        tag: str,
        method: str = "rmse"
    ) -> Dict[str, Path]:
        """
        Generiert separate PLY-Dateien für Outlier und Inlier.
        
        Returns:
            Dictionary mit Pfaden zu generierten Dateien
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        
        # Inlier PLY
        inlier_mask = ~outlier_mask
        if inlier_mask.any():
            inlier_path = output_dir / f"{tag}_inlier_{method}.ply"
            self.export_ply_with_scalar_field(
                points[inlier_mask],
                distances[inlier_mask],
                inlier_path,
                percentile_range=(5.0, 95.0)  # Engerer Range für Inlier
            )
            results['inlier'] = inlier_path
        
        # Outlier PLY
        if outlier_mask.any():
            outlier_path = output_dir / f"{tag}_outlier_{method}.ply"
            self.export_ply_with_scalar_field(
                points[outlier_mask],
                distances[outlier_mask],
                outlier_path,
                percentile_range=(0.0, 100.0)  # Voller Range für Outlier
            )
            results['outlier'] = outlier_path
        
        # Combined PLY mit unterschiedlichen Farben
        combined_path = output_dir / f"{tag}_outlier_marked_{method}.ply"
        self._export_outlier_marked_ply(
            points, distances, outlier_mask, combined_path
        )
        results['combined'] = combined_path
        
        logger.info(f"Generated outlier/inlier PLYs: {results}")
        return results
    
    def _export_outlier_marked_ply(
        self,
        points: np.ndarray,
        distances: np.ndarray,
        outlier_mask: np.ndarray,
        output_path: Path
    ) -> None:
        """Exportiert PLY mit markierten Outliern (rot) und Inliern (normal gefärbt)"""
        if not PLYFILE_AVAILABLE:
            return
        
        n = len(points)
        colors = np.zeros((n, 3), dtype=np.uint8)
        
        # Inlier: normale Distanz-Farben
        inlier_mask = ~outlier_mask
        if inlier_mask.any():
            inlier_colors = self._compute_distance_colors(
                distances, (128, 128, 128), (5.0, 95.0)
            )
            colors[inlier_mask] = inlier_colors[inlier_mask]
        
        # Outlier: Rot
        colors[outlier_mask] = [255, 0, 0]
        
        # Erstelle PLY mit beiden Scalar Fields
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('distance', 'f4'),
            ('is_outlier', 'u1')  # Zusätzliches Feld für Outlier-Status
        ]
        
        vertex = np.zeros(n, dtype=dtype)
        vertex['x'] = points[:, 0]
        vertex['y'] = points[:, 1]
        vertex['z'] = points[:, 2]
        vertex['red'] = colors[:, 0]
        vertex['green'] = colors[:, 1]
        vertex['blue'] = colors[:, 2]
        vertex['distance'] = distances
        vertex['is_outlier'] = outlier_mask.astype(np.uint8)
        
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=False).write(str(output_path))
    
    def save_coordinates_with_distances(
        self,
        coordinates: np.ndarray,
        distances: np.ndarray,
        output_path: Path,
        header: bool = True
    ) -> None:
        """
        Speichert Koordinaten mit Distanzen als TXT.
        Format: x y z distance
        """
        if len(coordinates) != len(distances):
            raise ValueError("Coordinates and distances must have same length")
        
        # Kombiniere Daten
        data = np.column_stack((coordinates, distances))
        
        # Speichere
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if header:
            np.savetxt(
                str(output_path),
                data,
                fmt='%.6f',
                header='x y z distance',
                comments=''
            )
        else:
            np.savetxt(str(output_path), data, fmt='%.6f')
        
        logger.info(f"Saved coordinates with distances: {output_path}")
    
    def create_m3c2_histogram(
        self,
        distances: np.ndarray,
        output_path: Path,
        title: str = "M3C2 Distance Distribution",
        bins: int = 256,
        figsize: Tuple[int, int] = (10, 6),
        show_statistics: bool = True
    ) -> None:
        """
        Erstellt M3C2-spezifisches Histogramm.
        
        Args:
            distances: M3C2 Distanzen
            output_path: Ausgabepfad
            title: Plot-Titel
            bins: Anzahl der Bins
            figsize: Figure-Größe
            show_statistics: Zeige Statistiken im Plot
        """
        # Filtere NaN-Werte
        valid_distances = distances[~np.isnan(distances)]
        
        if len(valid_distances) == 0:
            logger.warning("No valid distances for histogram")
            return
        
        # Erstelle Figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Histogramm
        n, bins_edges, patches = ax.hist(
            valid_distances,
            bins=bins,
            density=False,
            alpha=0.7,
            color='steelblue',
            edgecolor='black',
            linewidth=0.5
        )
        
        # Statistiken
        if show_statistics:
            mean = np.mean(valid_distances)
            std = np.std(valid_distances)
            median = np.median(valid_distances)
            
            # Vertikale Linien für Statistiken
            ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
            ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.4f}')
            ax.axvline(mean - std, color='orange', linestyle=':', linewidth=1, label=f'±1σ: {std:.4f}')
            ax.axvline(mean + std, color='orange', linestyle=':', linewidth=1)
            
            # Textbox mit Statistiken
            stats_text = (
                f'Count: {len(valid_distances)}\n'
                f'Mean: {mean:.4f}\n'
                f'Std: {std:.4f}\n'
                f'Min: {np.min(valid_distances):.4f}\n'
                f'Max: {np.max(valid_distances):.4f}'
            )
            ax.text(
                0.98, 0.97, stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        # Labels und Titel
        ax.set_xlabel('M3C2 Distance [m]', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Speichern
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created histogram: {output_path}")
    
    def create_comparison_histogram(
        self,
        data_dict: Dict[str, np.ndarray],
        output_path: Path,
        title: str = "Distance Comparison",
        bins: int = 50,
        alpha: float = 0.6
    ) -> None:
        """
        Erstellt überlagerte Histogramme für Vergleiche.
        
        Args:
            data_dict: Dictionary mit Label -> Distanzen
            output_path: Ausgabepfad
            title: Plot-Titel
            bins: Anzahl der Bins
            alpha: Transparenz
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(data_dict)))
        
        for i, (label, distances) in enumerate(data_dict.items()):
            valid = distances[~np.isnan(distances)]
            if len(valid) > 0:
                ax.hist(
                    valid,
                    bins=bins,
                    alpha=alpha,
                    label=f'{label} (n={len(valid)})',
                    color=colors[i],
                    edgecolor='black',
                    linewidth=0.5
                )
        
        ax.set_xlabel('M3C2 Distance [m]')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created comparison histogram: {output_path}")