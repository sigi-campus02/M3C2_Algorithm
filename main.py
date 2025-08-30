#!/usr/bin/env python3
# main.py
"""Haupteinstiegspunkt für M3C2 Pipeline - Clean Architecture Version"""

import sys
import logging
import argparse
from pathlib import Path
from typing import List

from shared.logging_setup import setup_logging
from application.factories.service_factory import ServiceFactory
from application.factories.pipeline_factory import PipelineFactory
from application.orchestration.pipeline_orchestrator import PipelineOrchestrator
from domain.builders.pipeline_builder import PipelineBuilder
from domain.entities import CloudPair, OutlierMethod, M3C2Parameters

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='M3C2 Pipeline - Refactored Clean Architecture',
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


def create_pipeline_configurations(
        cloud_pairs: List[CloudPair],
        args: argparse.Namespace,
        config: dict
) -> List:
    """
    Erstellt Pipeline-Konfigurationen aus CloudPairs und Argumenten.

    Args:
        cloud_pairs: Liste von CloudPair-Objekten
        args: Command-line Argumente
        config: Konfigurationsdictionary

    Returns:
        Liste von PipelineConfiguration-Objekten
    """
    configurations = []

    for cloud_pair in cloud_pairs:
        # Verwende Builder Pattern für saubere Konfiguration
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
            'output_base_path': Path(args.output_dir) / f"{args.project}_output",
            'generate_plots': args.plots
        })

        configurations.append(cfg.build())

    return configurations


def main():
    """
    Hauptfunktion - Orchestriert die gesamte Pipeline-Ausführung.

    Diese Funktion:
    1. Parst Command-line Argumente
    2. Lädt Konfiguration
    3. Scannt nach CloudPairs
    4. Erstellt Pipeline-Konfigurationen
    5. Führt Pipelines aus
    """
    # Parse Argumente
    args = parse_arguments()

    # Setup Logging
    log_file = args.log_file or 'logs/orchestration.log'
    setup_logging(
        level=args.log_level,
        log_file=log_file
    )

    logger.info("=" * 60)
    logger.info("M3C2 Pipeline - Clean Architecture")
    logger.info("=" * 60)

    # Lade Konfiguration
    from shared.config_loader import ConfigLoader
    config = ConfigLoader.load(args.config)
    config['data_path'] = args.data_dir
    config['output_path'] = args.output_dir

    # Initialisiere Service Factory
    service_factory = ServiceFactory(config)

    # Hole CloudPairScanner Service
    scanner = service_factory.get_cloud_pair_scanner()

    # Scanne nach CloudPairs
    base_path = Path(args.data_dir)
    folders = args.folders or config.get('folder_ids', ['Multi-Illumination'])

    all_cloud_pairs = []
    for folder_name in folders:
        folder_path = base_path / folder_name
        if not folder_path.exists():
            logger.warning(f"Folder not found: {folder_path}")
            continue

        cloud_pairs = scanner.scan_folder(folder_path, args.indices)
        all_cloud_pairs.extend(cloud_pairs)

    if not all_cloud_pairs:
        logger.error("No cloud pairs found to process")
        sys.exit(1)

    logger.info(f"Total cloud pairs to process: {len(all_cloud_pairs)}")

    # Erstelle Pipeline-Konfigurationen
    pipeline_configs = create_pipeline_configurations(all_cloud_pairs, args, config)
    logger.info(f"Created {len(pipeline_configs)} pipeline configurations")

    # Dry-Run Check
    if args.dry_run:
        logger.info("DRY RUN MODE - Showing what would be processed:")
        for i, cfg in enumerate(pipeline_configs, 1):
            logger.info(
                f"  [{i}] {cfg.cloud_pair.folder_id}: "
                f"{cfg.cloud_pair.tag} ({cfg.cloud_pair.comparison_case.value})"
            )
        logger.info("Exiting (dry run)")
        sys.exit(0)

    # Erstelle Pipeline Factory und Orchestrator
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