# application/orchestration/pipeline_orchestrator.py
class PipelineOrchestrator:
    """Koordiniert Pipeline-Ausführung"""
    
    def __init__(
        self,
        pipeline_factory: PipelineFactory,
        service_factory: ServiceFactory
    ):
        self.pipeline_factory = pipeline_factory
        self.service_factory = service_factory
    
    def run_batch(self, configurations: List[PipelineConfiguration]) -> None:
        for config in configurations:
            pipeline = self.pipeline_factory.create_pipeline(config)
            context = self._create_initial_context(config)
            
            try:
                result = pipeline.execute(context)
                self._save_results(result, config)
            except Exception as e:
                self._handle_error(e, config)
    
    def _create_initial_context(
        self, 
        config: PipelineConfiguration
    ) -> Dict[str, Any]:
        repository = self.service_factory.get_repository()
        
        return {
            'config': config,
            'moving_cloud': repository.load_point_cloud(config.moving_cloud),
            'reference_cloud': repository.load_point_cloud(config.reference_cloud),
            'm3c2_params': {
                'normal_scale': config.normal_scale,
                'search_scale': config.search_scale
            }
        }