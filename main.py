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
        '-p', '--project',
        type=str,
        default='default_project',
        help='Project name for outputs'
    )

    parser.add_argument(
        '--format',
        choices=['excel', 'json', 'csv'],
        default='excel',
        help='Output format for statistics'
    )

    # Logging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path'
    )

    # Andere
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without executing'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='M3C2 Pipeline v2.0.0 (Refactored)'
    )

    return parser.parse_args()


def scan_for_cloud_pairs(folder_path: Path, indices: Optional[List[int]] = None) -> List[CloudPair]:
    """
    Scannt einen Ordner nach Punktwolken-Paaren.

    Args:
        folder_path: Pfad zum Ordner
        indices: Spezifische Indizes (optional)

    Returns:
        Liste von CloudPair-Objekten
    """
    logger.info(f"Scanning folder: {folder_path.name}")

    # Finde alle Punktwolken-Dateien
    extensions = ['.ply', '.xyz', '.las', '.laz', '.txt']
    cloud_files = []

    for ext in extensions:
        cloud_files.extend(folder_path.glob(f"*{ext}"))

    # Parse Dateinamen
    a_plain = {}  # a-{index}
    a_ai = {}  # a-{index}-AI
    b_plain = {}  # b-{index}
    b_ai = {}  # b-{index}-AI

    for file in cloud_files:
        name = file.stem

        # Parse Index und Typ
        if name.startswith('a-') and not name.endswith('-AI'):
            try:
                index = int(name.split('-')[1])
                if indices is None or index in indices:
                    a_plain[index] = name
            except (IndexError, ValueError):
                continue

        elif name.startswith('a-') and name.endswith('-AI'):
            try:
                index = int(name.split('-')[1])
                if indices is None or index in indices:
                    a_ai[index] = name
            except (IndexError, ValueError):
                continue

        elif name.startswith('b-') and not name.endswith('-AI'):
            try:
                index = int(name.split('-')[1])
                if indices is None or index in indices:
                    b_plain[index] = name
            except (IndexError, ValueError):
                continue

        elif name.startswith('b-') and name.endswith('-AI'):
            try:
                index = int(name.split('-')[1])
                if indices is None or index in indices:
                    b_ai[index] = name
            except (IndexError, ValueError):
                continue

    # Log gefundene Dateien
    logger.info(f"  a_plain: {len(a_plain)} files (indices: {sorted(a_plain.keys())})")
    logger.info(f"  a_ai: {len(a_ai)} files (indices: {sorted(a_ai.keys())})")
    logger.info(f"  b_plain: {len(b_plain)} files (indices: {sorted(b_plain.keys())})")
    logger.info(f"  b_ai: {len(b_ai)} files (indices: {sorted(b_ai.keys())})")

    # Erstelle CloudPairs für alle möglichen Kombinationen
    cloud_pairs = []
    all_indices = set(a_plain.keys()) | set(a_ai.keys()) | set(b_plain.keys()) | set(b_ai.keys())

    for idx in sorted(all_indices):
        # Case 1: ai_vs_ai (a-i-AI vs b-i-AI)
        if idx in a_ai and idx in b_ai:
            cloud_pairs.append(CloudPair(
                moving_cloud=a_ai[idx],
                reference_cloud=b_ai[idx],
                folder_id=folder_path.name,
                comparison_case=ComparisonCase.CASE4,
                index=idx
            ))

        # Case 2: ai_vs_plain (a-i-AI vs b-i)
        if idx in a_ai and idx in b_plain:
            cloud_pairs.append(CloudPair(
                moving_cloud=a_ai[idx],
                reference_cloud=b_plain[idx],
                folder_id=folder_path.name,
                comparison_case=ComparisonCase.CASE3,
                index=idx
            ))

        # Case 3: plain_vs_ai (a-i vs b-i-AI)
        if idx in a_plain and idx in b_ai:
            cloud_pairs.append(CloudPair(
                moving_cloud=a_plain[idx],
                reference_cloud=b_ai[idx],
                folder_id=folder_path.name,
                comparison_case=ComparisonCase.CASE2,
                index=idx
            ))

        # Case 4: plain_vs_plain (a-i vs b-i)
        if idx in a_plain and idx in b_plain:
            cloud_pairs.append(CloudPair(
                moving_cloud=a_plain[idx],
                reference_cloud=b_plain[idx],
                folder_id=folder_path.name,
                comparison_case=ComparisonCase.CASE1,
                index=idx
            ))

    # Log Zusammenfassung
    case_counts = {}
    for pair in cloud_pairs:
        case = pair.comparison_case.value
        case_counts[case] = case_counts.get(case, 0) + 1

    logger.info(f"  Created {len(cloud_pairs)} cloud pairs")
    for case, count in sorted(case_counts.items()):
        logger.info(f"    {case}: {count} pairs")

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

    # Setup Logging
    log_file = args.log_file or 'logs/orchestration.log'
    setup_logging(
        log_level=getattr(logging, args.log_level),
        log_file=log_file
    )

    logger.info("=" * 60)
    logger.info("M3C2 Pipeline - Refactored Version")
    logger.info("=" * 60)

    # Lade Konfiguration
    from shared.config_loader import load_config
    config = load_config(args.config)
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
            parallel=False,  # Noch nicht implementiert
            continue_on_error=config.get('advanced', {}).get('continue_on_error', True)
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