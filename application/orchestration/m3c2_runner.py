# orchestration/m3c2_runner.py
"""M3C2 Runner - Kernimplementierung des M3C2-Algorithmus mit py4dgeo"""

import logging
import numpy as np
from typing import Tuple, Optional
import py4dgeo

logger = logging.getLogger(__name__)


class M3C2Runner:
    """
    Führt M3C2-Algorithmus mit py4dgeo aus.

    Diese Klasse kapselt die eigentliche M3C2-Berechnung und verwendet
    die py4dgeo-Bibliothek für die Implementierung.
    """

    def __init__(self):
        """Initialisiert den M3C2Runner"""
        self.last_computation_stats = None
        logger.debug("M3C2Runner initialized")

    def run(
            self,
            moving_cloud,
            reference_cloud,
            corepoints: np.ndarray,
            normal_scale: float,
            search_scale: float,
            max_depth: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Führt M3C2-Berechnung aus.

        Args:
            moving_cloud: Moving Point Cloud
            reference_cloud: Reference Point Cloud
            corepoints: Corepoints für die Berechnung (N x 3)
            normal_scale: Radius für Normalenschätzung
            search_scale: Suchradius für M3C2
            max_depth: Maximale Suchtiefe (optional)

        Returns:
            Tuple aus (distances, uncertainties) Arrays
        """
        logger.info(f"Starting M3C2 computation with {len(corepoints)} corepoints")
        logger.info(f"Parameters: normal_scale={normal_scale:.6f}, search_scale={search_scale:.6f}")

        try:
            # Konvertiere Punktwolken zu py4dgeo-Format
            epoch_mov = py4dgeo.Epoch(np.asarray(moving_cloud.cloud))
            epoch_ref = py4dgeo.Epoch(np.asarray(reference_cloud.cloud))

            # Erstelle M3C2-Objekt mit Parametern
            m3c2 = py4dgeo.M3C2(
                epochs=(epoch_ref, epoch_mov),
                corepoints=corepoints,
                normal_radii=normal_scale,
                cyl_radii=search_scale,
                max_distance=max_depth if max_depth else search_scale * 2
            )

            # Führe Berechnung aus
            distances, uncertainties = m3c2.run()

            # Berechne Statistiken
            valid_mask = ~np.isnan(distances)
            valid_count = np.sum(valid_mask)
            nan_count = len(distances) - valid_count

            self.last_computation_stats = {
                'total_points': len(distances),
                'valid_points': valid_count,
                'nan_points': nan_count,
                'nan_percentage': (nan_count / len(distances)) * 100,
                'mean_distance': np.nanmean(distances),
                'std_distance': np.nanstd(distances),
                'min_distance': np.nanmin(distances),
                'max_distance': np.nanmax(distances),
                'mean_uncertainty': np.nanmean(uncertainties)
            }

            logger.info(
                f"M3C2 completed: {valid_count}/{len(distances)} valid points "
                f"({self.last_computation_stats['nan_percentage']:.1f}% NaN)"
            )
            logger.info(
                f"Distance stats: mean={self.last_computation_stats['mean_distance']:.6f}, "
                f"std={self.last_computation_stats['std_distance']:.6f}"
            )

            return distances, uncertainties

        except Exception as e:
            logger.error(f"M3C2 computation failed: {str(e)}")
            raise RuntimeError(f"M3C2 computation failed: {str(e)}") from e

    def compute_with_subsampling(
            self,
            moving_cloud,
            reference_cloud,
            corepoints: np.ndarray,
            normal_scale: float,
            search_scale: float,
            subsample_factor: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Führt M3C2 mit optionalem Subsampling der Corepoints aus.

        Args:
            moving_cloud: Moving Point Cloud
            reference_cloud: Reference Point Cloud
            corepoints: Corepoints für die Berechnung
            normal_scale: Radius für Normalenschätzung
            search_scale: Suchradius für M3C2
            subsample_factor: Subsampling-Faktor (1 = kein Subsampling)

        Returns:
            Tuple aus (distances, uncertainties) Arrays
        """
        if subsample_factor > 1:
            logger.info(f"Subsampling corepoints by factor {subsample_factor}")
            subsampled_corepoints = corepoints[::subsample_factor]
            logger.info(f"Reduced from {len(corepoints)} to {len(subsampled_corepoints)} corepoints")
        else:
            subsampled_corepoints = corepoints

        return self.run(
            moving_cloud,
            reference_cloud,
            subsampled_corepoints,
            normal_scale,
            search_scale
        )

    def get_last_stats(self) -> Optional[dict]:
        """Gibt Statistiken der letzten Berechnung zurück"""
        return self.last_computation_stats.copy() if self.last_computation_stats else None