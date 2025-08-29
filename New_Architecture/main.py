# main.py
"""Main entry point für die M3C2 Pipeline-Anwendung"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Set, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import ConfigLoader
from shared.logging_setup import setup_logging
from application.factories.service_factory import ServiceFactory
from domain.entities import (
    PipelineConfiguration, 
    CloudPair, 
    ComparisonCase,
    ProcessingVersion,
    OutlierConfiguration,
    OutlierMethod,
    M3C2Parameters
)
from domain.builders.pipeline_builder import PipelineBuilder

logger = logging.getLogger(__name__)


class FileScanner:
    """Scannt Verzeichnisse nach Punktwolken-Dateien"""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
    
    def scan_folder(self, folder_id: str) -> dict:
        """Scannt einen Ordner nach Dateien im Pattern a-<i>[-AI] und b-<i>[-AI]"""
        import re
        
        folder_path = self.base_path / folder_id
        if not folder_path.exists():
            logger.warning(f"Folder not found: {folder_path}")
            return {}
        
        pattern = re.compile(r'^(?P<grp>[ab])-(?P<idx>\d+)(?P<ai>-AI)?$')
        
        # Kategorisiere Dateien
        files = {
            'a_plain': {},
            'a_ai': {},
            'b_plain': {},
            'b_ai': {}
        }
        
        for file_path in folder_path.iterdir():
            if not file_path.is_file():
                continue
            
            stem = file_path.stem
            match = pattern.match(stem)
            
            if match:
                group = match.group('grp')
                idx = int(match.group('idx'))
                has_ai = bool(match.group('ai'))
                
                category = f"{group}_{'ai' if has_ai else 'plain'}"
                files[category][idx] = stem
        
        return files
    
    def create_cloud_pairs(
        self, 
        folder_id: str, 
        files: dict,
        allowed_indices: Optional[Set[int]] = None
    ) -> List[CloudPair]:
        """Erstellt CloudPair-Objekte basierend auf gefundenen Dateien"""
        pairs = []
        
        # Case 1: a-i vs b-i (beide plain)
        common_plain = set(files['a_plain'].keys()) & set(files['b_plain'].keys())
        if allowed_indices:
            common_plain &= allowed_indices
        
        for idx in sorted(common_plain):
            pairs.append(CloudPair(
                moving_cloud=files['a_plain'][idx],
                reference_cloud=files['b_plain'][idx],
                folder_id=folder_id,
                comparison_case=ComparisonCase.CASE1,
                index=idx
            ))
        
        # Case 2: a-i vs b-i-AI (plain vs AI)
        common_case2 = set(files['a_plain'].keys()) & set(files['b_ai'].keys())
        if allowed_indices:
            common_case2 &= allowed_indices
        
        for idx in sorted(common_case2):
            pairs.append(CloudPair(
                moving_cloud=files['a_plain'][idx],
                reference_cloud=files['b_ai'][idx],
                folder_id=folder_id,
                comparison_case=ComparisonCase.CASE2,
                index=idx
            ))
        
        # Case 3: a-i-AI vs b-i (AI vs plain)
        common_case3 = set(files['a_ai'].keys()) & set(files['b_plain'].keys())
        if allowed_indices:
            common_case3 &= allowed_indices
        
        for idx in sorted(common_case3):
            pairs.append(CloudPair(
                moving_cloud=files['a_ai'][idx],
                reference_cloud=files['b_plain'][idx],
                folder_id=folder_id,
                comparison_case=ComparisonCase.CASE3,
                index=idx
            ))
        
        # Case 4: a-i-AI vs b-i-AI (beide AI)
        common_case4 = set(files['a_ai'].keys()) & set(files['b_ai'].keys())
        if allowed_indices:
            common_case4 &= allowed_indices
        
        for idx in sorted(common_case4):
            pairs.append(CloudPair(
                moving_cloud=files['a_ai'][idx],
                reference_cloud=files['b_ai'][idx],
                folder_id=folder_id,
                comparison_case=ComparisonCase.CASE4,
                index=idx
            ))
        
        return pairs


def parse_arguments() -> argparse.Namespace:
    """Parst Command-Line-Argumente"""
    parser = argparse.ArgumentParser(
        description="M3C2 Point Cloud Comparison Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Konfigurationsdatei
    parser.add_argument(
        '-c', '--config',
        type=str,
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
        version='%(prog)s 2.0.0 (Refactored)'
    )
    
    return parser.parse_args()


def build_configurations(
    config: dict,
    args: argparse.Namespace,
    cloud_pairs: List[CloudPair]
) -> List[PipelineConfiguration]:
    """Baut Pipeline-Konfigurationen aus Cloud-Paaren"""
    
    configurations = []
    builder = PipelineBuilder()
    
    for pair in cloud_pairs:
        # Basis-Konfiguration
        builder.reset()
        builder.with_cloud_pair(pair)
        builder.with_project(args.project)
        
        # M3C2 Parameter
        if args.normal_scale and args.search_scale:
            builder.with_m3c2_params(
                M3C2Parameters(args.normal_scale, args.search_scale)
            )
        elif not args.use_existing_params:
            # Auto-detection wird später durchgeführt
            pass
        
        # Outlier-Konfiguration
        outlier_method = OutlierMethod(args.outlier_method)
        builder.with_outlier_config(
            OutlierConfiguration(outlier_method, args.outlier_multiplier)
        )
        
        # Processing-Optionen
        builder.with_processing_options(
            use_existing_params=args.use_existing_params,
            only_stats=args.only_stats,
            mov_as_corepoints=config['processing'].get('mov_as_corepoints', True),
            use_subsampled_corepoints=config['processing'].get('use_subsampled_corepoints', 1)
        )
        
        # Output-Optionen
        builder.with_output_options(
            output_format=args.format,
            stats_type=config['processing'].get('stats_type', 'distance')
        )
        
        # Version
        version = ProcessingVersion(config['processing'].get('version', 'python'))
        builder.with_version(version)
        
        # Baue und validiere Konfiguration
        pipeline_config = builder.build()
        if pipeline_config.validate():
            configurations.append(pipeline_config)
        else:
            logger.warning(f"Invalid configuration for {pair.tag}, skipping")
    
    return configurations


def main():
    """Hauptfunktion"""
    
    # Parse Argumente
    args = parse_arguments()
    
    # Lade Konfiguration
    config = ConfigLoader.load(args.config)
    
    # Überschreibe Config mit CLI-Argumenten
    if args.data_dir:
        config['data_path'] = args.data_dir
    if args.output_dir:
        config['output_path'] = args.output_dir
    
    # Setup Logging
    setup_logging(
        config=config,
        log_file=args.log_file,
        level=args.log_level
    )
    
    logger.info("=" * 60)
    logger.info("M3C2 Pipeline - Refactored Version")
    logger.info("=" * 60)
    
    # Validiere Konfiguration
    if not ConfigLoader.validate(config):
        logger.error("Invalid configuration, exiting")
        sys.exit(1)
    
    # Bestimme zu verarbeitende Ordner
    if args.folders:
        folder_ids = args.folders
    else:
        folder_ids = config.get('folder_ids', [])
    
    if not folder_ids:
        logger.error("No folders specified for processing")
        sys.exit(1)
    
    # Scanne Dateien und erstelle Cloud-Paare
    scanner = FileScanner(Path(config['data_path']))
    all_pairs = []
    
    allowed_indices = set(args.indices) if args.indices else None
    
    for folder_id in folder_ids:
        logger.info(f"Scanning folder: {folder_id}")
        files = scanner.scan_folder(folder_id)
        
        # Log gefundene Dateien
        for category, items in files.items():
            if items:
                logger.info(f"  {category}: {len(items)} files (indices: {sorted(items.keys())})")
        
        # Erstelle Cloud-Paare
        pairs = scanner.create_cloud_pairs(folder_id, files, allowed_indices)
        logger.info(f"  Created {len(pairs)} cloud pairs")
        
        # Log Verteilung nach Cases
        case_counts = {}
        for pair in pairs:
            case = pair.comparison_case.value
            case_counts[case] = case_counts.get(case, 0) + 1
        
        for case, count in sorted(case_counts.items()):
            logger.info(f"    {case}: {count} pairs")
        
        all_pairs.extend(pairs)
    
    logger.info(f"Total cloud pairs to process: {len(all_pairs)}")
    
    if not all_pairs:
        logger.warning("No cloud pairs found to process")
        sys.exit(0)
    
    # Baue Pipeline-Konfigurationen
    configurations = build_configurations(config, args, all_pairs)
    logger.info(f"Created {len(configurations)} pipeline configurations")
    
    # Dry-Run Modus
    if args.dry_run:
        logger.info("DRY RUN MODE - Would process:")
        for i, cfg in enumerate(configurations, 1):
            logger.info(f"  {i}. {cfg.cloud_pair.folder_id}: {cfg.cloud_pair.tag} ({cfg.cloud_pair.comparison_case.value})")
        logger.info("Exiting (dry run)")
        sys.exit(0)
    
    # Erstelle Service Factory
    service_factory = ServiceFactory(config)
    
    # TODO: Hier würde der PipelineOrchestrator kommen
    logger.info("Ready to process - PipelineOrchestrator implementation pending")
    logger.warning("This is where the actual processing would happen")
    
    # Beispiel für Service-Nutzung
    pc_repo = service_factory.get_point_cloud_repository()
    param_repo = service_factory.get_parameter_repository()
    stats_repo = service_factory.get_statistics_repository()
    
    logger.info("Services initialized successfully")
    logger.info("Pipeline execution would start here...")
    
    # Cleanup
    service_factory.reset_services()
    logger.info("Pipeline completed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        sys.exit(1)