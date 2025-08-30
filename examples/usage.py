# New_Architecture/examples/usage.py
"""Bereinigte Beispiele für die neue Architektur"""

import numpy as np
from pathlib import Path

# Repository Imports
from infrastructure.repositories.point_cloud_repository import EnhancedPointCloudRepository
from infrastructure.repositories.distance_repository import DistanceRepository
from infrastructure.repositories.file_point_cloud_repository import FileParameterRepository

# Domain Entities
from domain.entities import (
    PipelineConfiguration, 
    CloudPair, 
    ComparisonCase,
    M3C2Parameters,
    OutlierConfiguration,
    OutlierMethod
)

# Commands
from domain.commands.m3c2_commands import (
    LoadPointCloudsCommand,
    RunM3C2Command,
    SaveResultsCommand,
    DetectOutliersCommand
)

from domain.commands.base import PipelineContext


def example_basic_repository_usage():
    """Grundlegende Verwendung der Repositories"""
    print("\n=== Basic Repository Usage ===")
    
    # Point Cloud Repository
    pc_repo = EnhancedPointCloudRepository("data")
    
    # Lade Punktwolke (Format wird automatisch erkannt)
    cloud = pc_repo.load_point_cloud("Multi-Illumination/a-1.ply")
    print(f"Loaded cloud with {cloud.cloud.shape[0]} points")
    
    # Konvertiere zu anderem Format
    pc_repo.convert_format(
        "Multi-Illumination/a-1.ply",
        "Multi-Illumination/a-1.xyz"
    )
    print("Converted PLY to XYZ format")
    
    # Distance Repository
    dist_repo = DistanceRepository("data")
    
    # Lade Distanzen
    distances = dist_repo.load_distances("Multi-Illumination/python_a-1-b-1_m3c2_distances.txt")
    print(f"Loaded {len(distances)} distances, {np.isnan(distances).sum()} NaN values")


def example_pipeline_with_commands():
    """Vollständige Pipeline mit Commands"""
    print("\n=== Pipeline with Commands ===")
    
    # Setup Repositories
    pc_repo = EnhancedPointCloudRepository("data")
    dist_repo = DistanceRepository("data")
    param_repo = FileParameterRepository("data")
    
    # Erstelle Konfiguration
    cloud_pair = CloudPair(
        moving_cloud="a-1",
        reference_cloud="b-1",
        folder_id="Multi-Illumination",
        comparison_case=ComparisonCase.CASE1,
        index=1
    )
    
    config = PipelineConfiguration(
        cloud_pair=cloud_pair,
        m3c2_params=M3C2Parameters(0.002, 0.004),
        outlier_config=OutlierConfiguration(
            method=OutlierMethod.RMSE,
            multiplier=3.0
        ),
        mov_as_corepoints=True,
        use_subsampled_corepoints=1
    )
    
    # Erstelle Context
    context = PipelineContext()
    context.set('config', config)
    
    # 1. Lade Punktwolken
    load_cmd = LoadPointCloudsCommand(pc_repo)
    if load_cmd.can_execute(context):
        context = load_cmd.execute(context)
        print("✓ Clouds loaded")
    
    # 2. M3C2 ausführen (Mock für Beispiel)
    # context = RunM3C2Command(m3c2_runner).execute(context)
    
    # Simuliere M3C2 Result für Beispiel
    from domain.entities import M3C2Result
    mock_distances = np.random.randn(1000) * 0.01
    mock_uncertainties = np.abs(np.random.randn(1000) * 0.001)
    
    result = M3C2Result.from_arrays(
        distances=mock_distances,
        uncertainties=mock_uncertainties,
        parameters=config.m3c2_params
    )
    context.set('m3c2_result', result)
    print("✓ M3C2 computed (simulated)")
    
    # 3. Outlier Detection
    from domain.strategies.outlier_detection import RMSEOutlierStrategy
    outlier_strategy = RMSEOutlierStrategy(multiplier=3.0)
    
    detect_cmd = DetectOutliersCommand(outlier_strategy)
    if detect_cmd.can_execute(context):
        context = detect_cmd.execute(context)
        stats = context.get('outlier_stats')
        print(f"✓ Outliers detected: {stats['outlier_count']}/{stats['total_points']}")
    
    # 4. Speichere Ergebnisse
    save_cmd = SaveResultsCommand(dist_repo, param_repo)
    if save_cmd.can_execute(context):
        context = save_cmd.execute(context)
        print("✓ Results saved")


def example_batch_processing():
    """Batch-Verarbeitung mehrerer Cloud-Paare"""
    print("\n=== Batch Processing ===")
    
    # Definiere mehrere Cloud-Paare
    pairs = [
        ("a-1", "b-1", ComparisonCase.CASE1),
        ("a-1", "b-1-AI", ComparisonCase.CASE2),
        ("a-1-AI", "b-1", ComparisonCase.CASE3),
        ("a-1-AI", "b-1-AI", ComparisonCase.CASE4),
    ]
    
    results = []
    
    for mov, ref, case in pairs:
        cloud_pair = CloudPair(
            moving_cloud=mov,
            reference_cloud=ref,
            folder_id="Multi-Illumination",
            comparison_case=case,
            index=1
        )
        
        print(f"Processing: {cloud_pair.tag} ({case.value})")
        
        # Hier würde die Pipeline ausgeführt werden
        # result = run_pipeline(cloud_pair)
        # results.append(result)
    
    print(f"Processed {len(pairs)} cloud pairs")


def example_format_handling():
    """Demonstration der Multi-Format Unterstützung"""
    print("\n=== Multi-Format Support ===")
    
    pc_repo = EnhancedPointCloudRepository("data")
    
    # Repository kann verschiedene Formate automatisch erkennen und laden
    formats = {
        "XYZ": "cloud.xyz",
        "PLY": "cloud.ply", 
        "LAS": "cloud.las",
        "LAZ": "cloud.laz",
        "OBJ": "cloud.obj"
    }
    
    for format_name, file_name in formats.items():
        # Simuliere Format-Erkennung
        if file_name.endswith('.xyz'):
            detected = pc_repo.detect_format(Path(file_name))
            print(f"{format_name}: detected as {detected}")
    
    # Konvertierung zwischen Formaten
    print("\nFormat conversions:")
    conversions = [
        ("input.ply", "output.xyz"),
        ("input.las", "output.ply"),
        ("input.obj", "output.xyz")
    ]
    
    for input_file, output_file in conversions:
        print(f"  {input_file} → {output_file}")
        # pc_repo.convert_format(input_file, output_file)


def example_statistics_workflow():
    """Statistik-Berechnung Workflow"""
    print("\n=== Statistics Workflow ===")
    
    dist_repo = DistanceRepository("data")
    
    # Lade Distanzen
    distances = np.random.randn(1000) * 0.01  # Simuliert
    distances[::10] = np.nan  # Einige NaN-Werte
    
    # Berechne Basis-Statistiken
    valid = distances[~np.isnan(distances)]
    
    stats = {
        'count': len(distances),
        'valid_count': len(valid),
        'nan_percentage': (np.isnan(distances).sum() / len(distances)) * 100,
        'mean': np.mean(valid),
        'std': np.std(valid),
        'rms': np.sqrt(np.mean(valid**2)),
        'min': np.min(valid),
        'max': np.max(valid),
        'median': np.median(valid)
    }
    
    print("Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    print("=" * 60)
    print("Clean Architecture Examples")
    print("=" * 60)
    
    # Führe alle Beispiele aus
    try:
        example_basic_repository_usage()
    except Exception as e:
        print(f"Error in basic usage: {e}")
    
    try:
        example_pipeline_with_commands()
    except Exception as e:
        print(f"Error in pipeline: {e}")
    
    try:
        example_batch_processing()
    except Exception as e:
        print(f"Error in batch processing: {e}")
    
    try:
        example_format_handling()
    except Exception as e:
        print(f"Error in format handling: {e}")
    
    try:
        example_statistics_workflow()
    except Exception as e:
        print(f"Error in statistics: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed")
    print("=" * 60)