# domain/strategies/outlier_detection.py
"""Strategien für Outlier Detection"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class OutlierDetectionStrategy(ABC):
    """Abstrakte Basis für Outlier-Detection Strategien"""
    
    def __init__(self, multiplier: float = 3.0):
        self.multiplier = multiplier
    
    @abstractmethod
    def detect(self, distances: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Erkennt Ausreißer in den Distanzen.
        
        Returns:
            Tuple[np.ndarray, float]: (Boolean-Array mit True=Outlier, Threshold-Wert)
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Gibt den Namen der Methode zurück"""
        pass
    
    def get_statistics(self, distances: np.ndarray, outliers: np.ndarray) -> dict:
        """Berechnet Statistiken über die Ausreißer"""
        total = len(distances)
        outlier_count = np.sum(outliers)
        inlier_count = total - outlier_count
        
        valid = distances[~np.isnan(distances)]
        outlier_values = distances[outliers & ~np.isnan(distances)]
        inlier_values = distances[~outliers & ~np.isnan(distances)]
        
        stats = {
            'total_points': total,
            'outlier_count': int(outlier_count),
            'inlier_count': int(inlier_count),
            'outlier_percentage': (outlier_count / total * 100) if total > 0 else 0,
            'method': self.get_method_name(),
            'multiplier': self.multiplier
        }
        
        if len(outlier_values) > 0:
            stats['outlier_mean'] = float(np.mean(outlier_values))
            stats['outlier_std'] = float(np.std(outlier_values))
            stats['outlier_max'] = float(np.max(np.abs(outlier_values)))
        
        if len(inlier_values) > 0:
            stats['inlier_mean'] = float(np.mean(inlier_values))
            stats['inlier_std'] = float(np.std(inlier_values))
            stats['inlier_max'] = float(np.max(np.abs(inlier_values)))
        
        return stats


class RMSEOutlierStrategy(OutlierDetectionStrategy):
    """RMSE-basierte Ausreißer-Erkennung"""
    
    def detect(self, distances: np.ndarray) -> Tuple[np.ndarray, float]:
        """Erkennt Ausreißer basierend auf RMSE"""
        valid = distances[~np.isnan(distances)]
        
        if len(valid) == 0:
            return np.zeros_like(distances, dtype=bool), 0.0
        
        # Berechne RMS (Root Mean Square)
        rms = np.sqrt(np.mean(valid ** 2))
        threshold = self.multiplier * rms
        
        # Markiere Ausreißer
        outliers = np.abs(distances) > threshold
        
        logger.info(
            f"RMSE outlier detection: RMS={rms:.6f}, "
            f"threshold={threshold:.6f}, "
            f"outliers={np.sum(outliers)}/{len(distances)}"
        )
        
        return outliers, threshold
    
    def get_method_name(self) -> str:
        return "RMSE"


class MADOutlierStrategy(OutlierDetectionStrategy):
    """MAD-basierte Ausreißer-Erkennung (Median Absolute Deviation)"""
    
    def detect(self, distances: np.ndarray) -> Tuple[np.ndarray, float]:
        """Erkennt Ausreißer basierend auf MAD"""
        valid = distances[~np.isnan(distances)]
        
        if len(valid) == 0:
            return np.zeros_like(distances, dtype=bool), 0.0
        
        # Berechne Median und MAD
        median = np.median(valid)
        mad = np.median(np.abs(valid - median))
        
        # Konvertiere MAD zu Standardabweichungs-Äquivalent
        # Faktor 1.4826 macht MAD konsistent mit Normalverteilung
        mad_std = 1.4826 * mad
        threshold = self.multiplier * mad_std
        
        # Markiere Ausreißer
        outliers = np.abs(distances - median) > threshold
        
        logger.info(
            f"MAD outlier detection: median={median:.6f}, "
            f"MAD={mad:.6f}, threshold={threshold:.6f}, "
            f"outliers={np.sum(outliers)}/{len(distances)}"
        )
        
        return outliers, threshold
    
    def get_method_name(self) -> str:
        return "MAD"


class IQROutlierStrategy(OutlierDetectionStrategy):
    """IQR-basierte Ausreißer-Erkennung (Interquartile Range)"""
    
    def __init__(self, multiplier: float = 1.5):
        # Standard IQR verwendet 1.5 als Multiplikator
        super().__init__(multiplier)
    
    def detect(self, distances: np.ndarray) -> Tuple[np.ndarray, float]:
        """Erkennt Ausreißer basierend auf IQR"""
        valid = distances[~np.isnan(distances)]
        
        if len(valid) == 0:
            return np.zeros_like(distances, dtype=bool), 0.0
        
        # Berechne Quartile
        q1 = np.percentile(valid, 25)
        q3 = np.percentile(valid, 75)
        iqr = q3 - q1
        
        # Definiere Grenzen
        lower_bound = q1 - self.multiplier * iqr
        upper_bound = q3 + self.multiplier * iqr
        
        # Markiere Ausreißer
        outliers = (distances < lower_bound) | (distances > upper_bound)
        
        # Threshold ist der maximale Abstand von den Grenzen
        threshold = max(abs(lower_bound), abs(upper_bound))
        
        logger.info(
            f"IQR outlier detection: Q1={q1:.6f}, Q3={q3:.6f}, "
            f"IQR={iqr:.6f}, bounds=[{lower_bound:.6f}, {upper_bound:.6f}], "
            f"outliers={np.sum(outliers)}/{len(distances)}"
        )
        
        return outliers, threshold
    
    def get_method_name(self) -> str:
        return "IQR"


class ZScoreOutlierStrategy(OutlierDetectionStrategy):
    """Z-Score-basierte Ausreißer-Erkennung"""
    
    def detect(self, distances: np.ndarray) -> Tuple[np.ndarray, float]:
        """Erkennt Ausreißer basierend auf Z-Score"""
        valid = distances[~np.isnan(distances)]
        
        if len(valid) == 0:
            return np.zeros_like(distances, dtype=bool), 0.0
        
        # Berechne Mittelwert und Standardabweichung
        mean = np.mean(valid)
        std = np.std(valid)
        
        if std == 0:
            # Alle Werte sind gleich
            return np.zeros_like(distances, dtype=bool), 0.0
        
        # Berechne Z-Scores
        z_scores = np.abs((distances - mean) / std)
        
        # Markiere Ausreißer
        outliers = z_scores > self.multiplier
        
        # Threshold in ursprünglichen Einheiten
        threshold = self.multiplier * std
        
        logger.info(
            f"Z-Score outlier detection: mean={mean:.6f}, "
            f"std={std:.6f}, threshold={threshold:.6f}, "
            f"outliers={np.sum(outliers)}/{len(distances)}"
        )
        
        return outliers, threshold
    
    def get_method_name(self) -> str:
        return "Z-Score"


class CombinedOutlierStrategy(OutlierDetectionStrategy):
    """Kombinierte Strategie - mehrere Methoden müssen übereinstimmen"""
    
    def __init__(self, strategies: list, min_agreement: int = 2):
        """
        Args:
            strategies: Liste von OutlierDetectionStrategy Instanzen
            min_agreement: Mindestanzahl von Strategien die übereinstimmen müssen
        """
        super().__init__(0)  # Multiplier wird von den einzelnen Strategien verwendet
        self.strategies = strategies
        self.min_agreement = min_agreement
    
    def detect(self, distances: np.ndarray) -> Tuple[np.ndarray, float]:
        """Kombiniert mehrere Strategien"""
        if not self.strategies:
            return np.zeros_like(distances, dtype=bool), 0.0
        
        # Sammle Ergebnisse aller Strategien
        outlier_votes = np.zeros_like(distances, dtype=int)
        thresholds = []
        
        for strategy in self.strategies:
            outliers, threshold = strategy.detect(distances)
            outlier_votes += outliers.astype(int)
            thresholds.append(threshold)
        
        # Markiere als Ausreißer wenn genug Strategien übereinstimmen
        combined_outliers = outlier_votes >= self.min_agreement
        
        # Durchschnittlicher Threshold
        avg_threshold = np.mean(thresholds)
        
        logger.info(
            f"Combined outlier detection: {len(self.strategies)} strategies, "
            f"min_agreement={self.min_agreement}, "
            f"outliers={np.sum(combined_outliers)}/{len(distances)}"
        )
        
        return combined_outliers, avg_threshold
    
    def get_method_name(self) -> str:
        strategy_names = [s.get_method_name() for s in self.strategies]
        return f"Combined({','.join(strategy_names)})"