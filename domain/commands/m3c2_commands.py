# domain/commands/m3c2_commands.py
"""M3C2 Pipeline Commands - Echte Implementierung"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional

from domain.commands.base import Command, PipelineContext
from domain.entities import M3C2Parameters, M3C2Result

logger = logging.getLogger(__name__)


class LoadPointCloudsCommand(Command):
    """Lädt Punktwolken aus Dateien"""

    def __init__(self, repository):
        super().__init__("LoadPointClouds")
        self.repository = repository

    def execute(self, context: PipelineContext) -> PipelineContext:
        self.log_execution(context)

        config = context.get('config')

        try:
            # Baue vollständige Pfade
            folder_path = Path(self.repository.base_path) / config.cloud_pair.folder_id
            mov_path = folder_path / config.cloud_pair.moving_cloud
            ref_path = folder_path / config.cloud_pair.reference_cloud

            logger.info(f"Loading moving cloud: {mov_path}")
            logger.info(f"Loading reference cloud: {ref_path}")

            # Lade Punktwolken mit Repository
            moving_cloud = self.repository.load(str(mov_path))
            reference_cloud = self.repository.load(str(ref_path))

            # Verwende moving cloud als Corepoints wenn konfiguriert
            if config.mov_as_corepoints:
                corepoints = moving_cloud
                logger.info(f"Using moving cloud as corepoints ({len(corepoints)} points)")
            else:
                # Subsample Corepoints
                subsample = config.use_subsampled_corepoints
                if subsample > 1:
                    corepoints = moving_cloud[::subsample]
                    logger.info(f"Subsampled corepoints to {len(corepoints)} points (factor {subsample})")
                else:
                    corepoints = moving_cloud

            context.set('moving_cloud', moving_cloud)
            context.set('reference_cloud', reference_cloud)
            context.set('corepoints', corepoints)

            logger.info(
                f"Loaded clouds - Moving: {len(moving_cloud)}, Reference: {len(reference_cloud)}, Corepoints: {len(corepoints)}")

        except Exception as e:
            self.handle_error(e, context)
            raise

        return context

    def can_execute(self, context: PipelineContext) -> bool:
        return context.has('config')


class EstimateParametersCommand(Command):
    """Schätzt M3C2-Parameter automatisch"""

    def __init__(self, param_estimator, strategy=None):
        super().__init__("EstimateParameters")
        self.param_estimator = param_estimator
        self.strategy = strategy

    def execute(self, context: PipelineContext) -> PipelineContext:
        self.log_execution(context)

        corepoints = context.get('corepoints')
        reference_cloud = context.get('reference_cloud')

        try:
            # Nutze auto_estimate für vollautomatische Schätzung
            params_dict = self.param_estimator.auto_estimate(
                corepoints=corepoints,
                reference_cloud=reference_cloud,
                sample_size=10000
            )

            # Erstelle M3C2Parameters Objekt
            params = M3C2Parameters(
                normal_scale=params_dict['normal_scale'],
                search_scale=params_dict['search_scale']
            )

            context.set('m3c2_params', params)
            context.set('param_estimation', params_dict)  # Speichere auch Details

            logger.info(
                f"Estimated parameters: normal={params.normal_scale:.6f}, "
                f"search={params.search_scale:.6f}, "
                f"avg_spacing={params_dict['avg_spacing']:.6f}"
            )

        except Exception as e:
            self.handle_error(e, context)
            raise

        return context

    def can_execute(self, context: PipelineContext) -> bool:
        return context.has('corepoints') and context.has('reference_cloud')


class RunM3C2Command(Command):
    """Führt M3C2-Algorithmus aus"""

    def __init__(self, m3c2_runner):
        super().__init__("RunM3C2")
        self.m3c2_runner = m3c2_runner

    def execute(self, context: PipelineContext) -> PipelineContext:
        self.log_execution(context)

        mov = context.get('moving_cloud')
        ref = context.get('reference_cloud')
        corepoints = context.get('corepoints')
        params = context.get('m3c2_params')

        try:
            # Führe M3C2 aus
            logger.info(f"Running M3C2 with {len(corepoints)} corepoints")

            distances, uncertainties = self.m3c2_runner.run(
                mov_cloud=mov,
                ref_cloud=ref,
                corepoints=corepoints,
                normal_scale=params.normal_scale,
                search_scale=params.search_scale
            )

            # Erstelle Result-Objekt
            result = M3C2Result.from_arrays(distances, uncertainties, params)
            context.set('m3c2_result', result)
            context.set('distances', distances)  # Für direkten Zugriff
            context.set('uncertainties', uncertainties)

            logger.info(
                f"M3C2 completed: {result.valid_count} valid points, "
                f"{result.nan_percentage:.1f}% NaN, "
                f"mean distance: {np.nanmean(distances):.6f}"
            )

        except Exception as e:
            self.handle_error(e, context)
            raise

        return context

    def can_execute(self, context: PipelineContext) -> bool:
        required = ['moving_cloud', 'reference_cloud', 'corepoints', 'm3c2_params']
        return all(context.has(key) for key in required)


class DetectOutliersCommand(Command):
    """Erkennt und markiert Ausreißer"""

    def __init__(self, outlier_strategy):
        super().__init__("DetectOutliers")
        self.outlier_strategy = outlier_strategy

    def execute(self, context: PipelineContext) -> PipelineContext:
        self.log_execution(context)

        result = context.get('m3c2_result')
        distances = result.distances

        try:
            # Filter NaN-Werte für Outlier-Detection
            valid_mask = ~np.isnan(distances)
            valid_distances = distances[valid_mask]

            if len(valid_distances) == 0:
                logger.warning("No valid distances for outlier detection")
                context.set('outliers', np.zeros_like(distances, dtype=bool))
                context.set('outlier_threshold', 0.0)
                return context

            # Verwende Outlier-Strategy
            outliers_valid, threshold = self.outlier_strategy.detect(valid_distances)

            # Mappe zurück auf volle Arrays (mit NaN)
            outliers = np.zeros_like(distances, dtype=bool)
            outliers[valid_mask] = outliers_valid

            context.set('outliers', outliers)
            context.set('outlier_threshold', threshold)
            context.set('inliers', ~outliers & valid_mask)

            n_outliers = np.sum(outliers)
            n_valid = np.sum(valid_mask)
            percentage = (n_outliers / n_valid * 100) if n_valid > 0 else 0

            logger.info(
                f"Detected {n_outliers} outliers ({percentage:.1f}% of valid points) "
                f"with threshold={threshold:.4f}"
            )

        except Exception as e:
            self.handle_error(e, context)
            raise

        return context

    def can_execute(self, context: PipelineContext) -> bool:
        return context.has('m3c2_result')


class ComputeStatisticsCommand(Command):
    """Berechnet umfangreiche Statistiken"""

    def __init__(self, statistics_service):
        super().__init__("ComputeStatistics")
        self.statistics_service = statistics_service

    def execute(self, context: PipelineContext) -> PipelineContext:
        self.log_execution(context)

        result = context.get('m3c2_result')
        outliers = context.get('outliers', None)
        config = context.get('config')

        try:
            # Berechne Statistiken mit Service
            stats = self.statistics_service.calculate_m3c2_statistics(
                distances=result.distances,
                uncertainties=result.uncertainties,
                outliers=outliers,
                outlier_config=config.outlier_config
            )

            # Füge Metadaten hinzu
            stats['folder_id'] = config.cloud_pair.folder_id
            stats['cloud_pair'] = config.cloud_pair.tag
            stats['comparison_case'] = config.cloud_pair.comparison_case.value
            stats['parameters_used'] = {
                'normal_scale': result.parameters_used.normal_scale,
                'search_scale': result.parameters_used.search_scale
            }

            context.set('statistics', stats)

            logger.info(
                f"Statistics computed - Mean: {stats.get('mean', 'N/A'):.6f}, "
                f"Std: {stats.get('std', 'N/A'):.6f}, "
                f"RMSE: {stats.get('rmse', 'N/A'):.6f}"
            )

        except Exception as e:
            self.handle_error(e, context)
            raise

        return context

    def can_execute(self, context: PipelineContext) -> bool:
        return context.has('m3c2_result')


class GenerateVisualizationCommand(Command):
    """Generiert Visualisierungen"""

    def __init__(self, visualization_service):
        super().__init__("GenerateVisualization")
        self.visualization_service = visualization_service

    def execute(self, context: PipelineContext) -> PipelineContext:
        self.log_execution(context)

        config = context.get('config')
        result = context.get('m3c2_result')

        # Überspringe wenn keine Plots gewünscht
        if not config.generate_plots:
            logger.info("Visualization generation skipped (generate_plots=False)")
            return context

        try:
            # Erstelle Output-Verzeichnis
            plots_dir = Path(config.output_base_path) / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Generiere Histogram
            hist_path = plots_dir / f"{config.cloud_pair.tag}_histogram.png"
            self.visualization_service.plot_histogram(
                distances=result.distances,
                output_path=str(hist_path),
                title=f"M3C2 Distances - {config.cloud_pair.tag}"
            )

            context.set('histogram_path', hist_path)
            logger.info(f"Generated histogram: {hist_path}")

            # Weitere Visualisierungen können hier hinzugefügt werden

        except Exception as e:
            # Visualisierung ist nicht kritisch - nur warnen
            logger.warning(f"Visualization failed: {e}")
            context.add_error(f"Visualization failed: {e}")

        return context

    def can_execute(self, context: PipelineContext) -> bool:
        return context.has('m3c2_result')


class SaveResultsCommand(Command):
    """Speichert Ergebnisse in verschiedenen Formaten"""

    def __init__(self, repository):
        super().__init__("SaveResults")
        self.repository = repository

    def execute(self, context: PipelineContext) -> PipelineContext:
        self.log_execution(context)

        config = context.get('config')
        stats = context.get('statistics', {})
        result = context.get('m3c2_result')

        try:
            # Erstelle Output-Verzeichnis
            output_dir = Path(config.output_base_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Speichere basierend auf Format
            if config.output_format == 'excel':
                output_path = output_dir / f"{config.cloud_pair.tag}_results.xlsx"
                self._save_to_excel(output_path, stats, result, config)
            elif config.output_format == 'csv':
                output_path = output_dir / f"{config.cloud_pair.tag}_results.csv"
                self._save_to_csv(output_path, stats, result, config)
            elif config.output_format == 'json':
                output_path = output_dir / f"{config.cloud_pair.tag}_results.json"
                self._save_to_json(output_path, stats, result, config)
            else:
                # Default: Excel
                output_path = output_dir / f"{config.cloud_pair.tag}_results.xlsx"
                self._save_to_excel(output_path, stats, result, config)

            context.set('output_path', output_path)
            logger.info(f"Results saved to: {output_path}")

        except Exception as e:
            self.handle_error(e, context)
            raise

        return context

    def _save_to_excel(self, path, stats, result, config):
        """Speichert in Excel-Format"""
        import pandas as pd

        # Erstelle DataFrame mit Statistiken
        stats_df = pd.DataFrame([stats])

        # Erstelle DataFrame mit Distanzen
        distances_df = pd.DataFrame({
            'distance': result.distances,
            'uncertainty': result.uncertainties
        })

        # Speichere in Excel mit mehreren Sheets
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            distances_df.to_excel(writer, sheet_name='Distances', index=False)

        logger.debug(f"Saved Excel with {len(stats_df)} stats and {len(distances_df)} distances")

    def _save_to_csv(self, path, stats, result, config):
        """Speichert in CSV-Format"""
        import pandas as pd

        # Kombiniere alles in einem DataFrame
        data = {
            'distance': result.distances,
            'uncertainty': result.uncertainties,
            **{f'stat_{k}': v for k, v in stats.items() if isinstance(v, (int, float))}
        }
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

        logger.debug(f"Saved CSV with {len(df)} rows")

    def _save_to_json(self, path, stats, result, config):
        """Speichert in JSON-Format"""
        import json
        import numpy as np

        # Konvertiere NumPy Arrays zu Listen
        data = {
            'statistics': stats,
            'distances': result.distances.tolist(),
            'uncertainties': result.uncertainties.tolist(),
            'config': {
                'cloud_pair': config.cloud_pair.tag,
                'normal_scale': result.parameters_used.normal_scale,
                'search_scale': result.parameters_used.search_scale
            }
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.debug(f"Saved JSON with statistics and {len(result.distances)} distances")

    def can_execute(self, context: PipelineContext) -> bool:
        return context.has('statistics')