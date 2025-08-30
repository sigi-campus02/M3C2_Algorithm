# application/services/param_estimator.py
"""Parameter Estimation Service"""

import logging
import numpy as np
from typing import Dict, Any, Optional

# IMPORTIERE die Strategies aus dem Domain Layer!
from domain.strategies.parameter_estimation import (
    ParameterEstimationStrategy,
    RadiusScanStrategy,
    AdaptiveStrategy
)

logger = logging.getLogger(__name__)


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