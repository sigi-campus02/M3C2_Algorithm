# main.py
from application.factories import ServiceFactory, PipelineFactory
from application.orchestration import PipelineOrchestrator
from infrastructure.config import ConfigLoader
from presentation.cli import CLIParser

def main():
    # Parse CLI arguments
    args = CLIParser().parse()
    
    # Load configuration
    config = ConfigLoader().load(args.config_file)
    
    # Setup dependencies
    service_factory = ServiceFactory(config)
    pipeline_factory = PipelineFactory(service_factory)
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        pipeline_factory,
        service_factory
    )
    
    # Build configurations
    configurations = build_configurations(
        config,
        args.filter_indices
    )
    
    # Run batch processing
    orchestrator.run_batch(configurations)

def build_configurations(
    config: Dict,
    filter_indices: Set[int] = None
) -> List[PipelineConfiguration]:
    """Erstellt Pipeline-Konfigurationen aus Eingabedaten"""
    
    builder = PipelineBuilder()
    configurations = []
    
    file_scanner = FileScanner(config['data_path'])
    file_groups = file_scanner.scan_and_group()
    
    for group in file_groups:
        if filter_indices and group.index not in filter_indices:
            continue
        
        config = (builder
            .with_data_source(group.folder)
            .with_clouds(group.moving, group.reference)
            .with_m3c2_params(
                normal_scale=config.get('normal_override'),
                search_scale=config.get('projection_override')
            )
            .with_outlier_detection(
                method=config['outlier_method'],
                multiplier=config['outlier_multiplier']
            )
            .build()
        )
        
        configurations.append(config)
    
    return configurations

if __name__ == "__main__":
    main()