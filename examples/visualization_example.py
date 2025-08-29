# New_Architecture/examples/visualization_example.py
"""Beispiel für die Nutzung der erweiterten Visualisierungs-Features"""

import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Imports
from application.factories.extended_service_factory import ExtendedServiceFactory
from application.factories.updated_pipeline_factory import UpdatedPipelineFactory
from application.orchestration.pipeline_orchestrator import PipelineOrchestrator
from domain.builders.pipeline_builder import PipelineBuilder
from domain.entities import CloudPair, PipelineContext


def example_visualization_pipeline():
    """Beispiel: Vollständige M3C2 Pipeline mit Visualisierungen"""
    
    # Konfiguration
    config = {
        'data_path': 'data',
        'output_path': 'outputs',
        'outlier_detection': {
            'method': 'rmse',
            'multiplier': 3.0
        },
        'plotting': {
            'bins': 256,
            'dpi': 100,
            'figure_size': [10, 6]
        }
    }
    
    # Services initialisieren
    service_factory = ExtendedServiceFactory(config)
    pipeline_factory = UpdatedPipelineFactory(service_factory)
    orchestrator = PipelineOrchestrator(service_factory)
    
    # Pipeline-Konfiguration erstellen
    builder = PipelineBuilder()
    pipeline_config = (
        builder
        .with_cloud_pair(CloudPair(
            mov='a-1.ply',
            ref='b-1.ply',
            tag='a-1-b-1'
        ))
        .with_folder_id('Multi-Illumination')
        .with_project_name('MARS_Visualization_Test')
        .with_output_format('excel')
        .with_outlier_config('rmse', 3.0)
        .build()
    )
    
    # Pipeline mit allen Visualisierungen erstellen
    commands = pipeline_factory.create_full_m3c2_pipeline(pipeline_config)
    
    logger.info(f"Pipeline enthält {len(commands)} Commands:")
    for cmd in commands:
        logger.info(f"  - {cmd.name}")
    
    # Pipeline ausführen
    result = orchestrator.run_single(pipeline_config)
    
    # Ergebnisse anzeigen
    if result.get('success'):
        logger.info("Pipeline erfolgreich ausgeführt!")
        
        # Zeige generierte Dateien
        if 'ply_exports' in result:
            logger.info("PLY Exports:")
            for key, path in result['ply_exports'].items():
                logger.info(f"  - {key}: {path}")
        
        if 'outlier_inlier_plys' in result:
            logger.info("Outlier/Inlier PLYs:")
            for key, path in result['outlier_inlier_plys'].items():
                logger.info(f"  - {key}: {path}")
        
        if 'coordinate_distance_files' in result:
            logger.info("Coordinate-Distance Files:")
            for key, path in result['coordinate_distance_files'].items():
                logger.info(f"  - {key}: {path}")
        
        if 'histogram_files' in result:
            logger.info("Histograms:")
            for key, path in result['histogram_files'].items():
                logger.info(f"  - {key}: {path}")
    else:
        logger.error("Pipeline fehlgeschlagen!")


def example_txt_to_ply_conversion():
    """Beispiel: Konvertiere bestehende TXT-Dateien zu PLY"""
    
    from application.services.enhanced_visualization_service import EnhancedVisualizationService
    
    viz_service = EnhancedVisualizationService()
    
    # Beispiel-Daten
    data_dir = Path('data/Multi-Illumination')
    
    # Finde alle TXT-Dateien mit Distanzen
    txt_files = list(data_dir.glob('*_m3c2_distances_coordinates*.txt'))
    
    logger.info(f"Gefundene TXT-Dateien: {len(txt_files)}")
    
    for txt_file in txt_files:
        try:
            # Konvertiere zu PLY
            ply_file = txt_file.with_suffix('.ply')
            viz_service.txt_to_ply_with_distance_color(
                txt_file,
                ply_file,
                scalar_name="m3c2_distance",
                percentile_range=(2.0, 98.0)
            )
            logger.info(f"Konvertiert: {txt_file.name} -> {ply_file.name}")
        except Exception as e:
            logger.error(f"Fehler bei {txt_file}: {e}")


def example_custom_visualization():
    """Beispiel: Benutzerdefinierte Visualisierung"""
    
    from application.services.enhanced_visualization_service import EnhancedVisualizationService
    
    viz_service = EnhancedVisualizationService()
    
    # Simulierte Daten
    n_points = 10000
    points = np.random.randn(n_points, 3) * 10
    distances = np.random.randn(n_points) * 0.01
    
    # Füge einige Outlier hinzu
    outlier_indices = np.random.choice(n_points, size=100, replace=False)
    distances[outlier_indices] *= 10
    
    # Erkenne Outlier (simple threshold)
    outlier_mask = np.abs(distances) > np.std(distances) * 3
    
    # Output-Verzeichnis
    output_dir = Path('outputs/custom_visualization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. PLY mit Scalar Field
    logger.info("Exportiere PLY mit Scalar Field...")
    viz_service.export_ply_with_scalar_field(
        points,
        distances,
        output_dir / 'points_with_distances.ply',
        scalar_name='distance',
        percentile_range=(5.0, 95.0)
    )
    
    # 2. Separate Outlier/Inlier PLYs
    logger.info("Generiere Outlier/Inlier PLYs...")
    results = viz_service.generate_outlier_inlier_plys(
        points,
        distances,
        outlier_mask,
        output_dir,
        'custom_data',
        'threshold'
    )
    
    # 3. Histogramm
    logger.info("Erstelle Histogramm...")
    viz_service.create_m3c2_histogram(
        distances,
        output_dir / 'distance_histogram.png',
        title='Custom Data - Distance Distribution',
        bins=100,
        show_statistics=True
    )
    
    # 4. Vergleichs-Histogramm
    logger.info("Erstelle Vergleichs-Histogramm...")
    comparison_data = {
        'All Points': distances,
        'Inliers': distances[~outlier_mask],
        'Outliers': distances[outlier_mask]
    }
    viz_service.create_comparison_histogram(
        comparison_data,
        output_dir / 'comparison_histogram.png',
        title='Inliers vs Outliers Comparison'
    )
    
    # 5. Speichere Koordinaten mit Distanzen
    logger.info("Speichere Koordinaten mit Distanzen...")
    viz_service.save_coordinates_with_distances(
        points,
        distances,
        output_dir / 'coordinates_with_distances.txt'
    )
    
    logger.info(f"Alle Visualisierungen gespeichert in: {output_dir}")


def example_batch_visualization():
    """Beispiel: Batch-Verarbeitung mit Visualisierungen"""
    
    config = {
        'data_path': 'data',
        'output_path': 'outputs/batch_viz',
        'outlier_detection': {'method': 'iqr', 'multiplier': 1.5}
    }
    
    # Services
    service_factory = ExtendedServiceFactory(config)
    pipeline_factory = UpdatedPipelineFactory(service_factory)
    orchestrator = PipelineOrchestrator(service_factory)
    
    # Mehrere Konfigurationen
    configs = []
    for i in range(1, 4):  # Part 1-3
        builder = PipelineBuilder()
        configs.append(
            builder
            .with_cloud_pair(CloudPair(f'a-{i}.ply', f'b-{i}.ply', f'a-{i}-b-{i}'))
            .with_folder_id('Multi-Illumination')
            .with_project_name('MARS_Batch_Viz')
            .build()
        )
    
    # Batch-Pipeline erstellen
    batch_commands = pipeline_factory.create_batch_visualization_pipeline(configs)
    
    logger.info(f"Batch-Pipeline mit {len(batch_commands)} Commands erstellt")
    
    # Batch ausführen
    results = orchestrator.run_batch(configs)
    
    logger.info(f"Batch-Verarbeitung abgeschlossen: {len(results)} Ergebnisse")


if __name__ == "__main__":
    import sys
    
    examples = {
        '1': ('Vollständige Pipeline mit Visualisierungen', example_visualization_pipeline),
        '2': ('TXT zu PLY Konvertierung', example_txt_to_ply_conversion),
        '3': ('Benutzerdefinierte Visualisierung', example_custom_visualization),
        '4': ('Batch-Visualisierung', example_batch_visualization)
    }
    
    print("Verfügbare Beispiele:")
    for key, (desc, _) in examples.items():
        print(f"  {key}: {desc}")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nWähle Beispiel (1-4): ")
    
    if choice in examples:
        print(f"\nStarte: {examples[choice][0]}")
        print("-" * 60)
        examples[choice][1]()
    else:
        print("Ungültige Auswahl")