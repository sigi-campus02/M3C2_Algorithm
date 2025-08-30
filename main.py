#!/usr/bin/env python3
# main.py
"""Haupteinstiegspunkt für M3C2 Pipeline - Refactored Version"""

import sys
import logging
import argparse
from pathlib import Path
from typing import List, Optional

from shared.logging_setup import setup_logging
from application.factories.service_factory import ServiceFactory
from application.factories.pipeline_factory import PipelineFactory
from application.orchestration.pipeline_orchestrator import PipelineOrchestrator
from domain.builders.pipeline_builder import PipelineBuilder
from domain.entities import CloudPair, ComparisonCase, OutlierMethod, M3C2Parameters

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='M3C2 Pipeline - Refactored Version',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Basis-Optionen
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config.json',
        help='Path to configuration file (JSON/YAML/INI)'
    )

    # Daten-Optionen
    parser.add_argument(
        '-d', '--data-dir',
        type=str,
        default='data',
        help='Base directory for data'
    )

    parser.add_argument(
        '-f', '--folders',
        nargs='+',
        help='Folder IDs to process'
    )

    parser.add_argument(
        '-i', '--indices',
        nargs='+',
        type=int,
        help='Specific indices to process (e.g., 1 2 3)'
    )

    # Processing-Optionen
    parser.add_argument(
        '--only-stats',
        action='store_true',
        help='Only compute statistics from existing results'
    )

    parser.add_argument(
        '--use-existing-params',
        action='store_true',
        help='Use existing M3C2 parameters if available'
    )

    parser.add_argument(
        '--normal-scale',
        type=float,
        help='Override normal scale parameter'
    )

    parser.add_argument(
        '--search-scale',
        type=float,
        help='Override search scale parameter'
    )

    # Outlier-Optionen
    parser.add_argument(
        '--outlier-method',
        choices=['rmse', 'mad', 'iqr', 'zscore'],
        default='rmse',
        help='Outlier detection method'
    )

    parser.add_argument(
        '--outlier-multiplier',
        type=float,
        default=3.0,
        help='Outlier detection multiplier'
    )

    # Output-Optionen
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='outputs',
        help='Output directory'
    )

    parser.add_argument(
        '--format',
        choices=['excel', 'csv', 'json', 'all'],
        default='excel',
        help='Output format'
    )

    parser.add_argument(
        '-p', '--project',
        type=str,
        default='M3C2_Results',
        help='Project name for outputs'
    )

    # Logging-Optionen
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path'
    )

    # Execution-Optionen
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without executing'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing'
    )

    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        default=True,
        help='Continue processing even if some pipelines fail'
    )

    # Visualisierung
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate visualization plots'
    )

    return parser.parse_args()


def scan_for_cloud_pairs(
        folder_path: Path,
        indices: Optional[List[int]] = None
) -> List[CloudPair]:
    """
    Scannt einen Ordner nach CloudPairs.

    Args:
        folder_path: Pfad zum Ordner
        indices: Optionale Liste von Indizes

    Returns:
        Liste von CloudPairs
    """
    cloud_pairs = []

    # Finde alle PLY-Dateien
    ply_files = list(folder_path.glob('*.ply'))

    if not ply_files:
        logger.warning(f"No PLY files found in {folder_path}")
        return cloud_pairs

    # Gruppiere nach Präfix
    groups = {}
    for file in ply_files:
        # Extrahiere Präfix (z.B. 'a_plain', 'b_ai')
        parts = file.stem.split('-')
        if len(parts) >= 2:
            prefix = parts[0]
            try:
                index = int(parts[1])
                if indices and index not in indices:
                    continue

                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append((index, file))
            except ValueError:
                logger.debug(f"Skipping file with non-numeric index: {file}")

    # Log gefundene Gruppen
    for prefix, files in groups.items():
        logger.info(f"  {prefix}: {len(files)} files (indices: {[f[0] for f in files]})")

    # Erstelle CloudPairs für alle Kombinationen
    prefixes = sorted(groups.keys())
    comparison_cases = {
        ('a_ai', 'b_ai'): ComparisonCase.AI_VS_AI,
        ('a_ai', 'b_plain'): ComparisonCase.AI_VS_PLAIN,
        ('a_plain', 'b_ai'): ComparisonCase.PLAIN_VS_AI,
        ('a_plain', 'b_plain'): ComparisonCase.PLAIN_VS_PLAIN,
    }

    # Zähler für Vergleichsfälle
    case_counts = {case: 0 for case in ComparisonCase}

    for i, prefix1 in enumerate(prefixes):
        for prefix2 in prefixes[i + 1:]:
            # Bestimme Vergleichsfall
            case_key = (prefix1, prefix2)
            if case_key not in comparison_cases:
                # Versuche umgekehrte Reihenfolge
                case_key = (prefix2, prefix1)

            if case_key not in comparison_cases:
                logger.debug(f"No comparison case for {prefix1} vs {prefix2}")
                continue

            comparison_case = comparison_cases[case_key]

            # Finde gemeinsame Indizes
            indices1 = {idx for idx, _ in groups[prefix1]}
            indices2 = {idx for idx, _ in groups[prefix2]}
            common_indices = indices1 & indices2

            # Erstelle Pairs für gemeinsame Indizes
            for idx in sorted(common_indices):
                file1 = next(f for i, f in groups[prefix1] if i == idx)
                file2 = next(f for i, f in groups[prefix2] if i == idx)

                # Bestimme mov/ref basierend auf Reihenfolge
                if case_key[0] == prefix1:
                    mov_file, ref_file = file1, file2
                else:
                    mov_file, ref_file = file2, file1

                cloud_pair = CloudPair(
                    mov=str(mov_file.name),
                    ref=str(ref_file.name),
                    tag=f"{mov_file.stem}-{ref_file.stem}",
                    folder_id=folder_path.name,
                    comparison_case=comparison_case
                )
                cloud_pairs.append(cloud_pair)
                case_counts[comparison_case] += 1

    # Log Zusammenfassung
    if cloud_pairs:
        logger.info(f"  Created {len(cloud_pairs)} cloud pairs")
        for case, count in case_counts.items():
            if count > 0:
                logger.info(f"    {case.value}: {count} pairs")

    return cloud_pairs


def create_pipeline_configurations(
        cloud_pairs: List[CloudPair],
        args: argparse.Namespace,
        config: dict
) -> List:
    """Erstellt Pipeline-Konfigurationen aus CloudPairs und Argumenten"""
    configurations = []
    builder = PipelineBuilder()

    for cloud_pair in cloud_pairs:
        # Reset Builder für jede Konfiguration
        builder = PipelineBuilder()

        # Basis-Konfiguration
        cfg = (
            builder
            .with_cloud_pair(cloud_pair)
            .with_folder_id(cloud_pair.folder_id)
            .with_project_name(args.project)
            .with_output_format(args.format)
            .with_outlier_config(
                OutlierMethod(args.outlier_method),
                args.outlier_multiplier
            )
        )

        # M3C2-Parameter wenn vorhanden
        if args.normal_scale and args.search_scale:
            cfg = cfg.with_m3c2_params(M3C2Parameters(
                normal_scale=args.normal_scale,
                search_scale=args.search_scale
            ))

        # Weitere Optionen
        cfg = cfg.with_options({
            'only_stats': args.only_stats,
            'use_existing_params': args.use_existing_params,
            'output_base_path': Path(args.output_dir) / f"{args.project}_output"
        })

        configurations.append(cfg.build())

    return configurations


def main():
    """Hauptfunktion"""
    # Parse Argumente
    args = parse_arguments()

    # Setup Logging - KORRIGIERT: level statt log_level
    log_file = args.log_file or 'logs/orchestration.log'
    setup_logging(
        level=args.log_level,  # Korrigiert von log_level zu level
        log_file=log_file
    )

    logger.info("=" * 60)
    logger.info("M3C2 Pipeline - Refactored Version")
    logger.info("=" * 60)

    # Lade Konfiguration
    from shared.config_loader import ConfigLoader
    config = ConfigLoader.load(args.config)
    config['data_path'] = args.data_dir
    config['output_path'] = args.output_dir

    # Scanne nach CloudPairs
    cloud_pairs = []
    folders = args.folders or config.get('folder_ids', ['Multi-Illumination'])

    for folder_name in folders:
        folder_path = Path(args.data_dir) / folder_name
        if not folder_path.exists():
            logger.warning(f"Folder not found: {folder_path}")
            continue

        logger.info(f"Scanning folder: {folder_name}")
        pairs = scan_for_cloud_pairs(folder_path, args.indices)
        cloud_pairs.extend(pairs)

    if not cloud_pairs:
        logger.error("No cloud pairs found to process")
        sys.exit(1)

    logger.info(f"Total cloud pairs to process: {len(cloud_pairs)}")

    # Erstelle Pipeline-Konfigurationen
    pipeline_configs = create_pipeline_configurations(cloud_pairs, args, config)
    logger.info(f"Created {len(pipeline_configs)} pipeline configurations")

    # Dry-Run Check
    if args.dry_run:
        logger.info("DRY RUN MODE - Showing what would be processed:")
        for i, cfg in enumerate(pipeline_configs, 1):
            logger.info(
                f"  [{i}] {cfg.cloud_pair.folder_id}: {cfg.cloud_pair.tag} ({cfg.cloud_pair.comparison_case.value})")
        logger.info("Exiting (dry run)")
        sys.exit(0)

    # Erstelle Service Factory und Orchestrator
    service_factory = ServiceFactory(config)
    pipeline_factory = PipelineFactory(service_factory)
    orchestrator = PipelineOrchestrator(service_factory, pipeline_factory)

    logger.info("Starting pipeline execution...")

    # Führe Pipelines aus
    try:
        orchestrator.run_batch(
            pipeline_configs,
            parallel=args.parallel,
            continue_on_error=args.continue_on_error
        )

        # Zeige Zusammenfassung
        results = orchestrator.get_results()
        failed = orchestrator.get_failed_configs()

        logger.info("=" * 60)
        logger.info("EXECUTION SUMMARY")
        logger.info(f"  Total: {len(pipeline_configs)}")
        logger.info(f"  Successful: {len(results)}")
        logger.info(f"  Failed: {len(failed)}")

        if failed:
            logger.warning("Failed configurations:")
            for cfg in failed:
                logger.warning(f"  - {cfg.cloud_pair.tag}")

        logger.info("Pipeline completed successfully!")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        service_factory.reset_services()
        logger.info("Services cleaned up")


if __name__ == "__main__":
    main()