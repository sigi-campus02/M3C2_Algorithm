from __future__ import annotations
import logging
import os
import time
from typing import List, Tuple
import numpy as np
from datasource.datasource import DataSource
from config.pipeline_config import PipelineConfig
from services.param_estimator import ParamEstimator
from services.statistics_service import StatisticsService
from services.exclude_outliers import exclude_outliers
from services.visualization_service import VisualizationService
from orchestration.m3c2_runner import M3C2Runner
from orchestration.strategies import (RadiusScanStrategy, ScaleScan)

logger = logging.getLogger(__name__)


class BatchOrchestrator:
    """Run the full M3C2 pipeline for a collection of configurations."""

    def __init__(
        self,
        configs: List[PipelineConfig],
        sample_size: int | None = None,
        output_format: str = "excel",
    ) -> None:
        self.configs = configs
        self.sample_size = sample_size
        self.output_format = output_format.lower()
        self.strategy = RadiusScanStrategy(sample_size=sample_size)

        logger.info("=== BatchOrchestrator initialisiert ===")
        logger.info("Konfigurationen: %d Jobs", len(self.configs))


    # Kleiner Helfer für konsistente Dateinamen
    @staticmethod
    def _run_tag(cfg: PipelineConfig) -> str:
        return f"{cfg.filename_mov}-{cfg.filename_ref}"
    
    def run_all(self) -> None:
        """Run the pipeline for each configured dataset."""
        if not self.configs:
            logger.warning("Keine Konfigurationen – nichts zu tun.")
            return

        for cfg in self.configs:
            try:
                self._run_single(cfg)
            except Exception:
                logger.exception("[Job] Fehler in Job '%s' (Version %s)", cfg.folder_id, cfg.filename_ref)

    def _run_single(self, cfg: PipelineConfig) -> None:
        logger.info(
            "%s, %s, %s, %s",
            cfg.folder_id,
            cfg.filename_mov,
            cfg.filename_ref,
            cfg.process_python_CC,
        )
        start = time.perf_counter()

        ds, mov, ref, corepoints = self._load_data(cfg)

        if cfg.process_python_CC == "python" and not cfg.only_stats:
            # Nur ausführen, wenn nicht nur Statistiken berechnet werden sollen
            out_base = ds.folder
            tag = self._run_tag(cfg)
            normal = projection = np.nan
            if cfg.use_existing_params:
                params_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_m3c2_params.txt")
                normal, projection = StatisticsService._load_params(params_path)

                if not np.isnan(normal) and not np.isnan(projection):
                    logger.info(
                        "[Params] geladen: %s (NormalScale=%.6f, SearchScale=%.6f)",
                        params_path,
                        normal,
                        projection,
                    )
                else:
                    logger.info(
                        "[Params] keine vorhandenen Parameter gefunden, berechne neu",
                    )
            if np.isnan(normal) or np.isnan(projection):
                normal, projection = self._determine_scales(cfg, corepoints)
                self._save_params(cfg, normal, projection, out_base)
            distances, _ = self._run_m3c2(
                cfg, mov, ref, corepoints, normal, projection, out_base
            )

            # all distances incl. outliers
            self._generate_visuals(cfg, mov, distances, out_base)

        try:
            # Generate distance txts excluding outliers & outliers only for outlier .ply visualisation
            logger.info("[Outlier] Entferne Ausreißer für %s", cfg.folder_id)
            self._exclude_outliers(cfg, ds.folder)
        except Exception:
            logger.exception("Fehler beim Entfernen von Ausreißern")

        try:
            logger.info("[Outlier] Erzeuge .ply Dateien für Outliers / Inliers …")
            self._generate_clouds_outliers(cfg, ds.folder)
        except Exception:
            logger.exception("Fehler beim Erzeugen von .ply Dateien für Ausreißer / Inlier")

        try:
            self._compute_statistics(cfg, ref)
        except Exception:
            logger.exception("Fehler bei der Berechnung der Statistik")

        logger.info("[Job] %s abgeschlossen in %.3fs", cfg.folder_id, time.perf_counter() - start)

    def _load_data(self, cfg: PipelineConfig) -> Tuple[DataSource, object, object, object]:
        t0 = time.perf_counter()
        ds = DataSource(
            cfg.folder_id,
            cfg.filename_mov,
            cfg.filename_ref,
            cfg.mov_as_corepoints,
            cfg.use_subsampled_corepoints
        )
        mov, ref, corepoints = ds.load_points()
        logger.info(
            "[Load] data/%s: mov=%s, ref=%s, corepoints=%s | %.3fs",
            cfg.folder_id,
            getattr(mov, "cloud", np.array([])).shape if hasattr(mov, "cloud") else "Epoch",
            getattr(ref, "cloud", np.array([])).shape if hasattr(ref, "cloud") else "Epoch",
            np.asarray(corepoints).shape,
            time.perf_counter() - t0,
        )
        return ds, mov, ref, corepoints

    def _determine_scales(self, cfg: PipelineConfig, corepoints) -> Tuple[float, float]:
        if cfg.normal_override is not None and cfg.proj_override is not None:
            normal, projection = cfg.normal_override, cfg.proj_override
            logger.info("[Scales] Overrides verwendet: normal=%.6f, proj=%.6f", normal, projection)
            return normal, projection

        t0 = time.perf_counter()
        avg = ParamEstimator.estimate_min_spacing(corepoints)
        logger.info("[Spacing] avg_spacing=%.6f (k=6) | %.3fs", avg, time.perf_counter() - t0)

        t0 = time.perf_counter()
        scans: List[ScaleScan] = ParamEstimator.scan_scales(corepoints, self.strategy, avg)
        logger.info("[Scan] %d Skalen evaluiert | %.3fs", len(scans), time.perf_counter() - t0)

        if scans:
            top_valid = sorted(scans, key=lambda s: s.valid_normals, reverse=True)[:5]
            logger.debug("  Top(valid_normals): %s", [(round(s.scale, 6), int(s.valid_normals)) for s in top_valid])
            top_smooth = sorted(scans, key=lambda s: (np.nan_to_num(s.roughness, nan=np.inf)))[:5]
            logger.debug("  Top(min_roughness): %s", [(round(s.scale, 6), float(s.roughness)) for s in top_smooth])

        t0 = time.perf_counter()
        normal, projection = ParamEstimator.select_scales(scans)
        logger.info("[Select] normal=%.6f, proj=%.6f | %.3fs", normal, projection, time.perf_counter() - t0)
        return normal, projection

    def _save_params(self, cfg: PipelineConfig, normal: float, projection: float, out_base: str) -> None:
        os.makedirs(out_base, exist_ok=True)
        tag = self._run_tag(cfg)
        params_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_m3c2_params.txt")
        with open(params_path, "w") as f:
            f.write(f"NormalScale={normal}\nSearchScale={projection}\n")
        logger.info("[Params] gespeichert: %s", params_path)

    def _run_m3c2(self, cfg: PipelineConfig, mov, ref, corepoints, normal: float, projection: float, out_base: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        tag = self._run_tag(cfg)

        t0 = time.perf_counter()
        runner = M3C2Runner()
        distances, uncertainties = runner.run(mov, ref, corepoints, normal, projection)
        duration = time.perf_counter() - t0
        n = len(distances)
        nan_share = float(np.isnan(distances).sum()) / n if n else 0.0
        logger.info("[Run] Punkte=%d | NaN=%.2f%% | Zeit=%.3fs", n, 100.0 * nan_share, duration)

        dists_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_m3c2_distances.txt")
        np.savetxt(dists_path, distances, fmt="%.6f")
        logger.info("[Run] Distanzen gespeichert: %s (%d Werte, %.2f%% NaN)", dists_path, n, 100.0 * nan_share)


        # NEU: Speichere XYZ-Koordinaten + Distanz in einer Datei


        coords_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_m3c2_distances_coordinates.txt")

        if hasattr(mov, "cloud"):
            xyz = np.asarray(mov.cloud)
        else:
            xyz = np.asarray(mov)
        if xyz.shape[0] == distances.shape[0]:
            arr = np.column_stack((xyz, distances))
            header = "x y z distance"
            np.savetxt(coords_path, arr, fmt="%.6f", header=header)
            logger.info(f"[Run] Distanzen mit Koordinaten gespeichert: {coords_path}")
        else:
            logger.warning(f"[Run] Anzahl Koordinaten stimmt nicht mit Distanzen überein: {xyz.shape[0]} vs {distances.shape[0]}")



        uncert_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_m3c2_uncertainties.txt")
        np.savetxt(uncert_path, uncertainties, fmt="%.6f")
        logger.info("[Run] Unsicherheiten gespeichert: %s", uncert_path)

        return distances, uncertainties

    def _exclude_outliers(self, cfg: PipelineConfig, out_base: str) -> None:
        tag = self._run_tag(cfg)
        exclude_outliers(
            data_folder=out_base,
            ref_variant=tag,
            method=cfg.outlier_detection_method,
            outlier_multiplicator=cfg.outlier_multiplicator
        )

    def _compute_statistics(self, cfg: PipelineConfig, ref) -> None:
        tag = self._run_tag(cfg)
        if cfg.stats_singleordistance == "distance":
            logger.info(f"[Stats on Distance] Berechne M3C2-Statistiken {cfg.folder_id},{cfg.filename_ref} …")

            if self.output_format == "excel":
                out_path = os.path.join(f"outputs/{cfg.project}_output/{cfg.project}_m3c2_stats_distances.xlsx")
            elif self.output_format == "json":
                out_path = os.path.join(f"outputs/{cfg.project}_output/{cfg.project}_m3c2_stats_distances.json")
            else:
                raise ValueError("Ungültiges Ausgabeformat. Verwenden Sie 'excel' oder 'json'.")

            StatisticsService.compute_m3c2_statistics(
                folder_ids=[cfg.folder_id],
                filename_ref=tag,
                process_python_CC=cfg.process_python_CC,
                out_path=out_path,
                sheet_name="Results",
                output_format=self.output_format,
                outlier_multiplicator=cfg.outlier_multiplicator,
                outlier_method=cfg.outlier_detection_method
            )

        if cfg.stats_singleordistance == "single":
            logger.info(
                f"[Stats on SingleClouds] Berechne M3C2-Statistiken {cfg.folder_id},{cfg.filename_ref} …"
            )

            if self.output_format == "excel":
                out_path = os.path.join(f"outputs/{cfg.project}_output/{cfg.project}_m3c2_stats_clouds.xlsx")
            elif self.output_format == "json":
                out_path = os.path.join(f"outputs/{cfg.project}_output/{cfg.project}_m3c2_stats_clouds.json")
            else:
                raise ValueError("Ungültiges Ausgabeformat. Verwenden Sie 'excel' oder 'json'.")

            StatisticsService.calc_single_cloud_stats(
                folder_ids=[cfg.folder_id],
                filename_mov=cfg.filename_mov,
                filename_ref=cfg.filename_ref,
                out_path=out_path,
                sheet_name="CloudStats",
                output_format=self.output_format
            )

    def _generate_visuals(self, cfg: PipelineConfig, mov, distances: np.ndarray, out_base: str) -> None:
        logger.info("[Visual] Erzeuge Visualisierungen …")
        tag = self._run_tag(cfg)
        os.makedirs(out_base, exist_ok=True)

        hist_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_histogram.png")
        ply_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_includenonvalid.ply")
        ply_valid_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}.ply")

        VisualizationService.histogram(distances, path=hist_path)
        logger.info("[Visual] Histogram gespeichert: %s", hist_path)

        colors = VisualizationService.colorize(mov.cloud, distances, outply=ply_path)
        logger.info("[Visual] Farb-PLY gespeichert: %s", ply_path)

        try:
            VisualizationService.export_valid(mov.cloud, colors, distances, outply=ply_valid_path)
            logger.info("[Visual] Valid-PLY gespeichert: %s", ply_valid_path)
        except Exception as exc:
            logger.warning("[Visual] Export valid-only übersprungen: %s", exc)

    def _generate_clouds_outliers(self, cfg: PipelineConfig, out_base: str) -> None:
        logger.info("[Visual] Erzeuge .ply Dateien für Outliers / Inliers …")
        os.makedirs(out_base, exist_ok=True)
        tag = self._run_tag(cfg)

        ply_valid_path_outlier = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_outlier_{cfg.outlier_detection_method}.ply")
        ply_valid_path_inlier = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_inlier_{cfg.outlier_detection_method}.ply")
        txt_path_outlier = os.path.join(out_base, f"python_{tag}_m3c2_distances_coordinates_outlier_{cfg.outlier_detection_method}.txt")
        txt_path_inlier = os.path.join(out_base, f"python_{tag}_m3c2_distances_coordinates_inlier_{cfg.outlier_detection_method}.txt")

        try:
            VisualizationService.txt_to_ply_with_distance_color(
                txt_path=txt_path_outlier,
                outply=ply_valid_path_outlier
            )
        except Exception as exc:
            logger.warning("[Visual] Export valid-only übersprungen: %s", exc)

        try:
            VisualizationService.txt_to_ply_with_distance_color(
                txt_path=txt_path_inlier,
                outply=ply_valid_path_inlier
            )
        except Exception as exc:
            logger.warning("[Visual] Export valid-only übersprungen: %s", exc)