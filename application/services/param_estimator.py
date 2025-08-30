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


# Fügen Sie dies am ENDE von application/services/param_estimator.py hinzu
# Nach der AdaptiveStrategy Klasse (etwa Zeile 143)

from typing import Optional  # Falls noch nicht importiert


class ParamEstimator:
    """Service für Parameter-Schätzung"""

    def __init__(self, default_strategy: Optional[ParameterEstimationStrategy] = None):
        """
        Initialisiert den ParamEstimator Service.

        Args:
            default_strategy: Standard-Strategy für Parameter-Schätzung
        """
        self.default_strategy = default_strategy or RadiusScanStrategy()
        logger.info("ParamEstimator service initialized")

    def auto_estimate(
            self,
            corepoints: np.ndarray,
            reference_cloud: np.ndarray,
            sample_size: int = 10000,
            strategy: Optional[ParameterEstimationStrategy] = None
    ) -> Dict[str, float]:
        """
        Schätzt M3C2-Parameter automatisch.

        Args:
            corepoints: Kernpunkte für M3C2
            reference_cloud: Referenz-Punktwolke
            sample_size: Anzahl der Punkte für Sampling
            strategy: Optionale Strategy (sonst default)

        Returns:
            Dictionary mit geschätzten Parametern
        """
        # Verwende angegebene oder Default-Strategy
        estimation_strategy = strategy or self.default_strategy

        logger.info(f"Starting parameter estimation with {estimation_strategy.__class__.__name__}")
        logger.info(f"Corepoints: {len(corepoints)}, Reference cloud: {len(reference_cloud)}")

        # Kombiniere Punkte für Schätzung oder nutze Referenz
        if len(corepoints) > 100:
            points_for_estimation = corepoints
        else:
            # Wenn zu wenige Corepoints, nutze Reference Cloud
            points_for_estimation = reference_cloud

        # Sample wenn zu viele Punkte
        if len(points_for_estimation) > sample_size:
            indices = np.random.choice(len(points_for_estimation), sample_size, replace=False)
            points_for_estimation = points_for_estimation[indices]
            logger.debug(f"Sampled {sample_size} points for parameter estimation")

        # Schätze Parameter mit Strategy
        params = estimation_strategy.estimate(points_for_estimation)

        # Validierung
        if params['normal_scale'] <= 0 or params['search_scale'] <= 0:
            logger.warning("Invalid parameters estimated, using defaults")
            params['normal_scale'] = 0.002
            params['search_scale'] = 0.004

        # Stelle sicher, dass search_scale >= normal_scale
        if params['search_scale'] < params['normal_scale']:
            params['search_scale'] = params['normal_scale'] * 2
            logger.debug(f"Adjusted search_scale to {params['search_scale']}")

        logger.info(
            f"Parameters estimated: normal_scale={params['normal_scale']:.6f}, "
            f"search_scale={params['search_scale']:.6f}"
        )

        return params

    def estimate_with_strategy(
            self,
            points: np.ndarray,
            strategy: ParameterEstimationStrategy
    ) -> Dict[str, float]:
        """
        Schätzt Parameter mit spezifischer Strategy.

        Args:
            points: Punktwolke
            strategy: Zu verwendende Strategy

        Returns:
            Dictionary mit geschätzten Parametern
        """
        logger.info(f"Estimating parameters with {strategy.__class__.__name__}")
        return strategy.estimate(points)

    def estimate_adaptive(
            self,
            points: np.ndarray,
            target_points: int = 30
    ) -> Dict[str, float]:
        """
        Schätzt Parameter mit adaptiver Strategy.

        Args:
            points: Punktwolke
            target_points: Ziel-Anzahl Punkte pro Nachbarschaft

        Returns:
            Dictionary mit geschätzten Parametern
        """
        strategy = AdaptiveStrategy(target_points=target_points)
        return self.estimate_with_strategy(points, strategy)

    def estimate_radius_scan(
            self,
            points: np.ndarray,
            sample_size: int = 10000
    ) -> Dict[str, float]:
        """
        Schätzt Parameter mit Radius-Scan Strategy.

        Args:
            points: Punktwolke
            sample_size: Sample-Größe für Schätzung

        Returns:
            Dictionary mit geschätzten Parametern
        """
        strategy = RadiusScanStrategy(sample_size=sample_size)
        return self.estimate_with_strategy(points, strategy)