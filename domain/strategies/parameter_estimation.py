# domain/strategies/parameter_estimation.py
"""Parameter Estimation Strategies"""

import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ParameterEstimationStrategy:
    """Basis-Klasse für Parameter-Estimation Strategien"""

    def estimate(self, points: np.ndarray, **kwargs) -> Dict[str, float]:
        """Schätzt Parameter basierend auf Punktwolke"""
        raise NotImplementedError


class RadiusScanStrategy(ParameterEstimationStrategy):
    """Strategy für Radius-basiertes Parameter-Scanning"""

    def __init__(self, sample_size: int = 10000):
        self.sample_size = sample_size
        self.scale_factors = [10, 15, 20, 25, 30, 35, 40]  # Multiplikatoren für avg_spacing
        logger.debug(f"RadiusScanStrategy initialized with sample_size={sample_size}")

    def estimate(self, points: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Schätzt optimale Radien durch Scanning verschiedener Werte.

        Args:
            points: Punktwolke
            **kwargs: Zusätzliche Parameter

        Returns:
            Dictionary mit geschätzten Parametern
        """
        # Sample Punkte wenn zu viele
        if len(points) > self.sample_size:
            indices = np.random.choice(len(points), self.sample_size, replace=False)
            sample_points = points[indices]
        else:
            sample_points = points

        # Berechne durchschnittlichen Punktabstand
        avg_spacing = self._estimate_avg_spacing(sample_points)

        # Teste verschiedene Scales
        best_normal = avg_spacing * 20  # Default
        best_search = best_normal * 2

        # Hier könnte eine elaboriertere Logik stehen
        # Für jetzt verwenden wir empirisch gute Werte

        return {
            'normal_scale': best_normal,
            'search_scale': best_search,
            'avg_spacing': avg_spacing
        }

    def _estimate_avg_spacing(self, points: np.ndarray) -> float:
        """Schätzt durchschnittlichen Punktabstand"""
        from sklearn.neighbors import NearestNeighbors

        # Finde nächste Nachbarn
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree')
        nbrs.fit(points)
        distances, _ = nbrs.kneighbors(points)

        # Durchschnitt der Abstände zum nächsten Nachbarn
        avg_spacing = np.mean(distances[:, 1])

        logger.debug(f"Estimated average point spacing: {avg_spacing:.6f}")
        return avg_spacing


class AdaptiveStrategy(ParameterEstimationStrategy):
    """Adaptive Strategy die sich an die lokale Punktdichte anpasst"""

    def __init__(self, target_points: int = 30):
        self.target_points = target_points
        logger.debug(f"AdaptiveStrategy initialized with target_points={target_points}")

    def estimate(self, points: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Passt Parameter adaptiv an die lokale Punktdichte an.

        Args:
            points: Punktwolke
            **kwargs: Zusätzliche Parameter

        Returns:
            Dictionary mit geschätzten Parametern
        """
        # Berechne lokale Dichte an verschiedenen Stellen
        densities = self._estimate_local_densities(points)

        # Wähle Parameter basierend auf durchschnittlicher Dichte
        avg_density = np.mean(densities)

        # Berechne Radius für Ziel-Punktanzahl
        volume_per_point = 1.0 / avg_density
        # Für eine Kugel: V = 4/3 * pi * r^3
        radius = np.cbrt(3 * self.target_points * volume_per_point / (4 * np.pi))

        return {
            'normal_scale': radius,
            'search_scale': radius * 2,
            'avg_density': avg_density
        }

    def _estimate_local_densities(self, points: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """Schätzt lokale Punktdichten"""
        from sklearn.neighbors import NearestNeighbors

        # Sample zufällige Punkte
        n_samples = min(n_samples, len(points))
        sample_indices = np.random.choice(len(points), n_samples, replace=False)
        sample_points = points[sample_indices]

        # Finde k nächste Nachbarn für jeden Sample-Punkt
        k = min(30, len(points))
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
        nbrs.fit(points)
        distances, _ = nbrs.kneighbors(sample_points)

        # Berechne Dichte als Punkte pro Volumen
        # Volumen einer Kugel mit Radius = maximale Distanz
        max_distances = distances[:, -1]
        volumes = (4 / 3) * np.pi * max_distances ** 3
        densities = k / volumes

        return densities