# New_Architecture/domain/strategies/statistics_strategies.py
"""Strategy Pattern für verschiedene Statistik-Berechnungen"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, normaltest, kstest, jarque_bera
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


class StatisticsStrategy(ABC):
    """Abstrakte Basis für Statistik-Strategien"""
    
    @abstractmethod
    def calculate(self, data: np.ndarray) -> Dict[str, float]:
        """Berechnet Statistiken"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Gibt den Namen der Strategy zurück"""
        pass


class BasicStatisticsStrategy(StatisticsStrategy):
    """Strategy für Basis-Statistiken"""
    
    def calculate(self, data: np.ndarray) -> Dict[str, float]:
        """Berechnet Basis-Statistiken"""
        if len(data) == 0:
            return self._empty_stats()
        
        return {
            'count': len(data),
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data)),
            'variance': float(np.var(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'range': float(np.max(data) - np.min(data)),
            'sum': float(np.sum(data))
        }
    
    def _empty_stats(self) -> Dict[str, float]:
        """Gibt leere Statistiken zurück"""
        return {
            'count': 0,
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'variance': 0.0,
            'min': 0.0,
            'max': 0.0,
            'range': 0.0,
            'sum': 0.0
        }
    
    def get_name(self) -> str:
        return "basic_statistics"


class AdvancedStatisticsStrategy(StatisticsStrategy):
    """Strategy für erweiterte Statistiken"""
    
    def calculate(self, data: np.ndarray) -> Dict[str, float]:
        """Berechnet erweiterte Statistiken"""
        if len(data) == 0:
            return {}
        
        basic = BasicStatisticsStrategy().calculate(data)
        
        # Percentiles
        percentiles = {
            'p1': float(np.percentile(data, 1)),
            'p5': float(np.percentile(data, 5)),
            'p10': float(np.percentile(data, 10)),
            'q1': float(np.percentile(data, 25)),
            'q3': float(np.percentile(data, 75)),
            'p90': float(np.percentile(data, 90)),
            'p95': float(np.percentile(data, 95)),
            'p99': float(np.percentile(data, 99))
        }
        
        # IQR and outlier bounds
        q1, q3 = percentiles['q1'], percentiles['q3']
        iqr = q3 - q1
        
        outlier_stats = {
            'iqr': float(iqr),
            'lower_fence': float(q1 - 1.5 * iqr),
            'upper_fence': float(q3 + 1.5 * iqr),
            'outlier_count': int(np.sum((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)))
        }
        
        # Moments
        moments = {
            'skewness': float(stats.skew(data)),
            'kurtosis': float(stats.kurtosis(data)),
            'cv': float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else 0.0
        }
        
        return {**basic, **percentiles, **outlier_stats, **moments}
    
    def get_name(self) -> str:
        return "advanced_statistics"


class DistanceStatisticsStrategy(StatisticsStrategy):
    """Strategy für Distanz-spezifische Statistiken"""
    
    def calculate(self, data: np.ndarray) -> Dict[str, float]:
        """Berechnet Distanz-Statistiken"""
        if len(data) == 0:
            return {}
        
        # Basic stats
        basic = BasicStatisticsStrategy().calculate(data)
        
        # Distance-specific metrics
        distance_stats = {
            'rmse': float(np.sqrt(np.mean(data**2))),
            'mae': float(np.mean(np.abs(data))),
            'mse': float(np.mean(data**2)),
            'std_error': float(np.std(data) / np.sqrt(len(data))),
            'ci95_lower': float(np.mean(data) - 1.96 * np.std(data) / np.sqrt(len(data))),
            'ci95_upper': float(np.mean(data) + 1.96 * np.std(data) / np.sqrt(len(data)))
        }
        
        # Absolute statistics
        abs_data = np.abs(data)
        abs_stats = {
            'abs_mean': float(np.mean(abs_data)),
            'abs_median': float(np.median(abs_data)),
            'abs_std': float(np.std(abs_data)),
            'abs_max': float(np.max(abs_data))
        }
        
        # Signed statistics (positive vs negative)
        pos_data = data[data > 0]
        neg_data = data[data < 0]
        
        signed_stats = {
            'positive_count': len(pos_data),
            'negative_count': len(neg_data),
            'positive_mean': float(np.mean(pos_data)) if len(pos_data) > 0 else 0.0,
            'negative_mean': float(np.mean(neg_data)) if len(neg_data) > 0 else 0.0,
            'balance_ratio': len(pos_data) / len(data) if len(data) > 0 else 0.5
        }
        
        return {**basic, **distance_stats, **abs_stats, **signed_stats}
    
    def get_name(self) -> str:
        return "distance_statistics"


class NormalityTestStrategy(StatisticsStrategy):
    """Strategy für Normalverteilungs-Tests"""
    
    def calculate(self, data: np.ndarray) -> Dict[str, float]:
        """Führt Normalverteilungs-Tests durch"""
        if len(data) < 3:
            return {'normality_tests': 'insufficient_data'}
        
        results = {}
        
        # Shapiro-Wilk test (best for small samples)
        if len(data) <= 5000:
            statistic, p_value = shapiro(data)
            results['shapiro_statistic'] = float(statistic)
            results['shapiro_pvalue'] = float(p_value)
            results['shapiro_normal'] = p_value > 0.05
        
        # D'Agostino-Pearson test
        if len(data) >= 8:
            statistic, p_value = normaltest(data)
            results['dagostino_statistic'] = float(statistic)
            results['dagostino_pvalue'] = float(p_value)
            results['dagostino_normal'] = p_value > 0.05
        
        # Kolmogorov-Smirnov test
        statistic, p_value = kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        results['ks_statistic'] = float(statistic)
        results['ks_pvalue'] = float(p_value)
        results['ks_normal'] = p_value > 0.05
        
        # Jarque-Bera test
        if len(data) >= 2000:
            statistic, p_value = jarque_bera(data)
            results['jb_statistic'] = float(statistic)
            results['jb_pvalue'] = float(p_value)
            results['jb_normal'] = p_value > 0.05
        
        # Overall assessment
        normal_tests = [v for k, v in results.items() if k.endswith('_normal')]
        if normal_tests:
            results['is_normal'] = sum(normal_tests) / len(normal_tests) > 0.5
        
        return results
    
    def get_name(self) -> str:
        return "normality_tests"


class RobustStatisticsStrategy(StatisticsStrategy):
    """Strategy für robuste Statistiken"""
    
    def calculate(self, data: np.ndarray) -> Dict[str, float]:
        """Berechnet robuste Statistiken"""
        if len(data) == 0:
            return {}
        
        # Robust location measures
        location = {
            'trimmed_mean_10': float(stats.trim_mean(data, 0.1)),
            'trimmed_mean_20': float(stats.trim_mean(data, 0.2)),
            'winsorized_mean_10': float(stats.mstats.winsorize(data, limits=[0.1, 0.1]).mean()),
            'median': float(np.median(data)),
            'mode': float(stats.mode(data, keepdims=False)[0]) if len(data) > 0 else 0.0
        }
        
        # Robust scale measures
        scale = {
            'mad': float(stats.median_abs_deviation(data)),
            'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
            'gini_mad': float(self._gini_mean_difference(data)),
            'qn_scale': float(self._qn_scale(data)),
            'sn_scale': float(self._sn_scale(data))
        }
        
        # Robust skewness and kurtosis
        shape = {
            'medcouple': float(self._medcouple(data)),
            'tail_weight': float(self._tail_weight(data))
        }
        
        return {**location, **scale, **shape}
    
    def _gini_mean_difference(self, data: np.ndarray) -> float:
        """Berechnet Gini mean difference"""
        n = len(data)
        if n < 2:
            return 0.0
        
        sorted_data = np.sort(data)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_data) / (n * np.sum(sorted_data))) - (n + 1) / n
    
    def _qn_scale(self, data: np.ndarray) -> float:
        """Berechnet Qn scale estimator"""
        n = len(data)
        if n < 2:
            return 0.0
        
        # Pairwise differences
        diffs = []
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(abs(data[i] - data[j]))
        
        # Return first quartile
        return np.percentile(diffs, 25) * 2.2219
    
    def _sn_scale(self, data: np.ndarray) -> float:
        """Berechnet Sn scale estimator"""
        n = len(data)
        if n < 2:
            return 0.0
        
        medians = []
        for i in range(n):
            diffs = np.abs(data - data[i])
            medians.append(np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 0)
        
        return np.median(medians) * 1.1926
    
    def _medcouple(self, data: np.ndarray) -> float:
        """Berechnet Medcouple (robuste Schiefe)"""
        n = len(data)
        if n < 3:
            return 0.0
        
        med = np.median(data)
        data_centered = data - med
        
        left = data_centered[data_centered <= 0]
        right = data_centered[data_centered >= 0]
        
        if len(left) == 0 or len(right) == 0:
            return 0.0
        
        h_values = []
        for l in left:
            for r in right:
                if l != -r:
                    h = (r + l) / (r - l)
                    h_values.append(h)
        
        return np.median(h_values) if h_values else 0.0
    
    def _tail_weight(self, data: np.ndarray) -> float:
        """Berechnet Tail Weight (robuste Kurtosis)"""
        q = np.percentile(data, [12.5, 25, 75, 87.5])
        
        if q[2] - q[1] == 0:
            return 0.0
        
        return (q[3] - q[0]) / (q[2] - q[1])
    
    def get_name(self) -> str:
        return "robust_statistics"


class ComparisonStatisticsStrategy(StatisticsStrategy):
    """Strategy für Vergleichs-Statistiken zwischen zwei Datensätzen"""
    
    def calculate(self, data: Any) -> Dict[str, float]:
        """Berechnet Vergleichs-Statistiken"""
        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError("Comparison requires tuple of two datasets")
        
        data1, data2 = data
        
        if len(data1) == 0 or len(data2) == 0:
            return {}
        
        # Ensure same length for paired comparisons
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        # Basic comparisons
        basic = {
            'mean_diff': float(np.mean(data1) - np.mean(data2)),
            'median_diff': float(np.median(data1) - np.median(data2)),
            'std_diff': float(np.std(data1) - np.std(data2)),
            'mean_abs_diff': float(np.mean(np.abs(data1 - data2)))
        }
        
        # Correlation measures
        correlation = {
            'pearson_r': float(np.corrcoef(data1, data2)[0, 1]),
            'spearman_r': float(stats.spearmanr(data1, data2)[0]),
            'kendall_tau': float(stats.kendalltau(data1, data2)[0])
        }
        
        # Agreement measures
        agreement = {
            'icc': float(self._calculate_icc(data1, data2)),
            'concordance': float(self._calculate_concordance(data1, data2)),
            'cohens_d': float(self._calculate_cohens_d(data1, data2))
        }
        
        # Statistical tests
        tests = {}
        
        # T-test
        t_stat, t_pval = stats.ttest_rel(data1, data2)
        tests['paired_t_statistic'] = float(t_stat)
        tests['paired_t_pvalue'] = float(t_pval)
        tests['significant_diff'] = t_pval < 0.05
        
        # Wilcoxon signed-rank test
        if len(data1) >= 20:
            w_stat, w_pval = stats.wilcoxon(data1, data2)
            tests['wilcoxon_statistic'] = float(w_stat)
            tests['wilcoxon_pvalue'] = float(w_pval)
        
        return {**basic, **correlation, **agreement, **tests}
    
    def _calculate_icc(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Berechnet Intraclass Correlation Coefficient"""
        n = len(data1)
        
        # Reshape for ICC calculation
        Y = np.array([data1, data2]).T
        
        # Mean squares
        grand_mean = np.mean(Y)
        between_ms = n * np.sum((np.mean(Y, axis=1) - grand_mean) ** 2) / (n - 1)
        within_ms = np.sum((Y - np.mean(Y, axis=1, keepdims=True)) ** 2) / n
        
        if between_ms + within_ms == 0:
            return 0.0
        
        return (between_ms - within_ms) / (between_ms + within_ms)
    
    def _calculate_concordance(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Berechnet Lin's Concordance Correlation Coefficient"""
        mean1, mean2 = np.mean(data1), np.mean(data2)
        var1, var2 = np.var(data1), np.var(data2)
        
        covariance = np.mean((data1 - mean1) * (data2 - mean2))
        
        numerator = 2 * covariance
        denominator = var1 + var2 + (mean1 - mean2) ** 2
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_cohens_d(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Berechnet Cohen's d effect size"""
        mean_diff = np.mean(data1) - np.mean(data2)
        pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
        
        if pooled_std == 0:
            return 0.0
        
        return mean_diff / pooled_std
    
    def get_name(self) -> str:
        return "comparison_statistics"


class TimeSeriesStatisticsStrategy(StatisticsStrategy):
    """Strategy für Zeitreihen-Statistiken"""
    
    def calculate(self, data: np.ndarray) -> Dict[str, float]:
        """Berechnet Zeitreihen-Statistiken"""
        if len(data) < 2:
            return {}
        
        # Trend analysis
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        trend = {
            'trend_slope': float(slope),
            'trend_intercept': float(intercept),
            'trend_r2': float(r_value ** 2),
            'trend_pvalue': float(p_value),
            'trend_std_error': float(std_err)
        }
        
        # Autocorrelation
        autocorr = {
            'autocorr_lag1': float(self._autocorrelation(data, 1)),
            'autocorr_lag2': float(self._autocorrelation(data, 2)),
            'autocorr_lag5': float(self._autocorrelation(data, 5))
        }
        
        # Stationarity
        diff_data = np.diff(data)
        stationarity = {
            'mean_change': float(np.mean(diff_data)),
            'variance_change': float(np.var(diff_data)),
            'max_change': float(np.max(np.abs(diff_data)))
        }
        
        # Volatility
        volatility = {
            'rolling_std_5': float(pd.Series(data).rolling(5).std().mean()),
            'rolling_std_10': float(pd.Series(data).rolling(10).std().mean()) if len(data) >= 10 else 0.0
        }
        
        return {**trend, **autocorr, **stationarity, **volatility}
    
    def _autocorrelation(self, data: np.ndarray, lag: int) -> float:
        """Berechnet Autokorrelation für gegebenen Lag"""
        if lag >= len(data):
            return 0.0
        
        c0 = np.var(data)
        if c0 == 0:
            return 0.0
        
        c_lag = np.cov(data[:-lag], data[lag:])[0, 1]
        return c_lag / c0
    
    def get_name(self) -> str:
        return "timeseries_statistics"


# Factory für Statistik-Strategien
class StatisticsStrategyFactory:
    """Factory für Statistik-Strategien"""
    
    _strategies = {
        'basic': BasicStatisticsStrategy,
        'advanced': AdvancedStatisticsStrategy,
        'distance': DistanceStatisticsStrategy,
        'normality': NormalityTestStrategy,
        'robust': RobustStatisticsStrategy,
        'comparison': ComparisonStatisticsStrategy,
        'timeseries': TimeSeriesStatisticsStrategy
    }
    
    @classmethod
    def create_strategy(cls, strategy_type: str) -> StatisticsStrategy:
        """Erstellt eine Statistik-Strategy"""
        strategy_class = cls._strategies.get(strategy_type.lower())
        if not strategy_class:
            raise ValueError(f"Unknown statistics strategy: {strategy_type}")
        
        return strategy_class()
    
    @classmethod
    def create_multiple(cls, strategy_types: List[str]) -> List[StatisticsStrategy]:
        """Erstellt mehrere Strategien"""
        return [cls.create_strategy(s) for s in strategy_types]
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type) -> None:
        """Registriert eine neue Strategy"""
        if not issubclass(strategy_class, StatisticsStrategy):
            raise ValueError("Strategy must inherit from StatisticsStrategy")
        
        cls._strategies[name.lower()] = strategy_class
        logger.info(f"Registered statistics strategy: {name}")
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Gibt verfügbare Strategien zurück"""
        return list(cls._strategies.keys())


# Aggregator für mehrere Statistik-Strategien
class StatisticsAggregator:
    """Aggregiert Ergebnisse mehrerer Statistik-Strategien"""
    
    def __init__(self, strategies: List[StatisticsStrategy] = None):
        self.strategies = strategies or [
            BasicStatisticsStrategy(),
            AdvancedStatisticsStrategy()
        ]
    
    def calculate_all(self, data: np.ndarray) -> Dict[str, Any]:
        """Berechnet alle Statistiken"""
        results = {}
        
        for strategy in self.strategies:
            try:
                strategy_results = strategy.calculate(data)
                results[strategy.get_name()] = strategy_results
            except Exception as e:
                logger.error(f"Error in {strategy.get_name()}: {e}")
                results[strategy.get_name()] = {'error': str(e)}
        
        # Add combined summary
        results['summary'] = self._create_summary(results)
        
        return results
    
    def _create_summary(self, results: Dict) -> Dict:
        """Erstellt eine Zusammenfassung"""
        summary = {
            'strategies_used': len(self.strategies),
            'successful_calculations': sum(1 for v in results.values() if 'error' not in v)
        }
        
        # Extract key metrics
        if 'basic_statistics' in results:
            basic = results['basic_statistics']
            summary.update({
                'mean': basic.get('mean', 0),
                'std': basic.get('std', 0),
                'count': basic.get('count', 0)
            })
        
        return summary