# services/param_estimator.py
"""Parameter Estimator für automatische M3C2-Parameter-Schätzung"""

import logging
import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScaleEvaluation:
    """Ergebnis einer Scale-Evaluierung"""
    normal_scale: float
    search_scale: float
    score: float
    valid_ratio: float
    mean_points_in_cylinder: float


class ParamEstimator:
    """
    Schätzt optimale M3C2-Parameter basierend auf Punktwolken-Eigenschaften.

    Diese Klasse analysiert die Punktdichte und räumliche Verteilung,
    um geeignete Normal- und Search-Scales zu bestimmen.
    """

    def __init__(self):
        """Initialisiert den ParamEstimator"""
        self.min_points_for_normal = 30  # Minimum Punkte für Normalenschätzung
        self.min_points_in_cylinder = 10  # Minimum Punkte im Zylinder
        self.scale_multipliers = [10, 15, 20, 25, 30, 35, 40]  # Multiplikatoren für Scales
        logger.debug("ParamEstimator initialized")

    def estimate_min_spacing(
            self,
            points: np.ndarray,
            sample_size: int = 10000,
            k_neighbors: int = 2
    ) -> float:
        """
        Schätzt den durchschnittlichen minimalen Punktabstand.

        Args:
            points: Punktwolke (N x 3)
            sample_size: Anzahl Sample-Punkte für Schätzung
            k_neighbors: Anzahl nächster Nachbarn

        Returns:
            Durchschnittlicher minimaler Punktabstand
        """
        logger.info(f"Estimating point spacing from {len(points)} points")

        # Sample wenn zu viele Punkte
        if len(points) > sample_size:
            indices = np.random.choice(len(points), sample_size, replace=False)
            sample_points = points[indices]
        else:
            sample_points = points

        # Finde nächste Nachbarn
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree')
        nbrs.fit(sample_points)
        distances, _ = nbrs.kneighbors(sample_points)

        # Berechne durchschnittlichen Abstand zum nächsten Nachbarn
        # (Index 1, da Index 0 der Punkt selbst ist)
        avg_spacing = np.mean(distances[:, 1])

        logger.info(f"Average point spacing: {avg_spacing:.6f}")
        return avg_spacing

    def scan_scales(
            self,
            corepoints: np.ndarray,
            reference_cloud: np.ndarray,
            avg_spacing: float
    ) -> List[ScaleEvaluation]:
        """
        Testet verschiedene Scale-Kombinationen.

        Args:
            corepoints: Corepoints für Evaluation
            reference_cloud: Referenz-Punktwolke
            avg_spacing: Durchschnittlicher Punktabstand

        Returns:
            Liste von ScaleEvaluation-Objekten
        """
        evaluations = []

        # Sample Corepoints für schnellere Evaluation
        sample_size = min(1000, len(corepoints))
        sample_indices = np.random.choice(len(corepoints), sample_size, replace=False)
        sample_corepoints = corepoints[sample_indices]

        logger.info(f"Scanning scales with {sample_size} sample corepoints")

        # Erstelle KD-Tree für effiziente Nachbarschaftssuche
        from sklearn.neighbors import KDTree
        tree = KDTree(reference_cloud)

        for multiplier in self.scale_multipliers:
            normal_scale = avg_spacing * multiplier
            search_scale = normal_scale * 2  # Search Scale ist typisch 2x Normal Scale

            # Evaluiere diese Kombination
            valid_count = 0
            total_points_in_cylinders = []

            for corepoint in sample_corepoints:
                # Finde Punkte im Normal-Radius
                points_in_normal = tree.query_radius(
                    corepoint.reshape(1, -1),
                    r=normal_scale
                )[0]

                # Finde Punkte im Search-Radius
                points_in_search = tree.query_radius(
                    corepoint.reshape(1, -1),
                    r=search_scale
                )[0]

                # Prüfe ob genug Punkte für Berechnung
                if len(points_in_normal) >= self.min_points_for_normal:
                    valid_count += 1
                    total_points_in_cylinders.append(len(points_in_search))

            valid_ratio = valid_count / len(sample_corepoints)
            mean_points = np.mean(total_points_in_cylinders) if total_points_in_cylinders else 0

            # Berechne Score (höher ist besser)
            # Wir wollen: hohe Valid-Ratio, moderate Punktanzahl
            score = valid_ratio * min(1.0, mean_points / 100)  # Normalisiere auf ~100 Punkte

            evaluation = ScaleEvaluation(
                normal_scale=normal_scale,
                search_scale=search_scale,
                score=score,
                valid_ratio=valid_ratio,
                mean_points_in_cylinder=mean_points
            )
            evaluations.append(evaluation)

            logger.debug(
                f"Scale {multiplier}x: normal={normal_scale:.6f}, "
                f"search={search_scale:.6f}, valid={valid_ratio:.2%}, "
                f"points={mean_points:.1f}, score={score:.3f}"
            )

        return evaluations

    def select_scales(
            self,
            evaluations: List[ScaleEvaluation],
            min_valid_ratio: float = 0.8
    ) -> Tuple[float, float]:
        """
        Wählt die besten Scales basierend auf Evaluierungen.

        Args:
            evaluations: Liste von ScaleEvaluation-Objekten
            min_valid_ratio: Minimale Valid-Ratio für Akzeptanz

        Returns:
            Tuple aus (normal_scale, search_scale)
        """
        # Filtere nach Mindest-Valid-Ratio
        valid_evaluations = [
            e for e in evaluations
            if e.valid_ratio >= min_valid_ratio
        ]

        if not valid_evaluations:
            logger.warning(
                f"No scales with valid_ratio >= {min_valid_ratio}, "
                "using best available"
            )
            valid_evaluations = evaluations

        # Wähle beste nach Score
        best = max(valid_evaluations, key=lambda e: e.score)

        logger.info(
            f"Selected scales: normal={best.normal_scale:.6f}, "
            f"search={best.search_scale:.6f} "
            f"(valid={best.valid_ratio:.2%}, score={best.score:.3f})"
        )

        return best.normal_scale, best.search_scale

    def auto_estimate(
            self,
            corepoints: np.ndarray,
            reference_cloud: np.ndarray,
            sample_size: int = 10000
    ) -> Dict[str, float]:
        """
        Vollautomatische Parameter-Schätzung.

        Args:
            corepoints: Corepoints
            reference_cloud: Referenz-Punktwolke
            sample_size: Sample-Größe für Schätzung

        Returns:
            Dictionary mit geschätzten Parametern
        """
        logger.info("Starting automatic parameter estimation")

        # 1. Schätze Punktabstand
        avg_spacing = self.estimate_min_spacing(reference_cloud, sample_size)

        # 2. Scanne verschiedene Scales
        evaluations = self.scan_scales(corepoints, reference_cloud, avg_spacing)

        # 3. Wähle beste Parameter
        normal_scale, search_scale = self.select_scales(evaluations)

        return {
            'normal_scale': normal_scale,
            'search_scale': search_scale,
            'avg_spacing': avg_spacing,
            'evaluation_count': len(evaluations)
        }