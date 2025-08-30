# application/services/statistics_service.py
"""Wrapper für ModularStatisticsService zur Kompatibilität"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from domain.strategies.statistics_strategies import StatisticsStrategyFactory, StatisticsAggregator

logger = logging.getLogger(__name__)


class StatisticsService:
    """Service für statistische Berechnungen - Wrapper für ModularStatisticsService"""

    def __init__(self):
        """Initialisiert den Statistics Service mit Strategies"""
        self.strategy_factory = StatisticsStrategyFactory()
        self.default_strategies = [
            self.strategy_factory.create_strategy('basic'),
            self.strategy_factory.create_strategy('distance'),
            self.strategy_factory.create_strategy('advanced')
        ]
        self.aggregator = StatisticsAggregator(self.default_strategies)
        logger.info("StatisticsService initialized (using strategy pattern)")

    def calculate_m3c2_statistics(
            self,
            distances: np.ndarray,
            uncertainties: Optional[np.ndarray] = None,
            outliers: Optional[np.ndarray] = None,
            outlier_config: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Berechnet umfassende Statistiken für M3C2 Ergebnisse.
        Nutzt die vorhandenen Statistics Strategies.
        """
        stats = {}

        # Entferne NaN und Inf Werte
        valid_mask = np.isfinite(distances)
        valid_distances = distances[valid_mask]

        if len(valid_distances) == 0:
            logger.warning("No valid distances found")
            return {'error': 'No valid distances'}

        # Nutze Aggregator für umfassende Statistiken
        all_stats = self.aggregator.calculate_all(valid_distances)

        # Flatten die Ergebnisse
        for strategy_name, strategy_results in all_stats.items():
            if isinstance(strategy_results, dict) and 'error' not in strategy_results:
                if strategy_name == 'basic_statistics':
                    # Kopiere Basis-Statistiken auf oberste Ebene
                    stats.update(strategy_results)
                elif strategy_name == 'distance_statistics':
                    # Füge Distance-spezifische Metriken hinzu
                    stats.update(strategy_results)
                else:
                    # Andere Strategien als nested dict
                    stats[strategy_name] = strategy_results

        # Outlier-Statistiken
        if outliers is not None:
            outlier_mask = outliers & valid_mask
            inlier_mask = ~outliers & valid_mask

            stats['outlier_count'] = int(np.sum(outlier_mask))
            stats['inlier_count'] = int(np.sum(inlier_mask))
            stats['outlier_percentage'] = (stats['outlier_count'] / len(valid_distances) * 100
                                           if len(valid_distances) > 0 else 0)

            # Statistiken nur für Inliers
            if np.any(inlier_mask):
                inliers = distances[inlier_mask]
                inlier_stats = self.aggregator.calculate_all(inliers)

                # Extrahiere wichtigste Inlier-Statistiken
                if 'basic_statistics' in inlier_stats:
                    basic = inlier_stats['basic_statistics']
                    stats['inliers_only'] = {
                        'mean': basic.get('mean', 0),
                        'median': basic.get('median', 0),
                        'std': basic.get('std', 0),
                        'rmse': inlier_stats.get('distance_statistics', {}).get('rmse', 0),
                        'mae': inlier_stats.get('distance_statistics', {}).get('mae', 0),
                        'count': basic.get('count', 0)
                    }
            else:
                stats['inliers_only'] = {'count': 0}

        # Unsicherheits-Statistiken
        if uncertainties is not None:
            valid_uncertainties = uncertainties[valid_mask]
            if len(valid_uncertainties) > 0:
                uncertainty_stats = self.strategy_factory.create_strategy('basic').calculate(valid_uncertainties)
                stats['uncertainty'] = {
                    'mean': uncertainty_stats.get('mean', 0),
                    'median': uncertainty_stats.get('median', 0),
                    'std': uncertainty_stats.get('std', 0),
                    'min': uncertainty_stats.get('min', 0),
                    'max': uncertainty_stats.get('max', 0)
                }

        # Füge Gesamt-Counts hinzu
        stats['total_count'] = len(distances)
        stats['valid_count'] = len(valid_distances)
        stats['invalid_count'] = stats['total_count'] - stats['valid_count']

        return stats

    def calculate_batch_statistics(self, batch_results: List[Dict[str, Any]]) -> Any:
        """Delegiert an den ModularStatisticsService wenn nötig"""
        # Simplified implementation
        import pandas as pd

        if not batch_results:
            return pd.DataFrame()

        stats_list = []
        for result in batch_results:
            if 'statistics' in result:
                stats = result['statistics'].copy()
                stats['cloud_pair'] = result.get('cloud_pair', '')
                stats_list.append(stats)

        return pd.DataFrame(stats_list) if stats_list else pd.DataFrame()