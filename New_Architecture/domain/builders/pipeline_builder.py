# domain/builders/pipeline_builder.py
class PipelineBuilder:
    """Fluent Interface für Pipeline-Konfiguration"""
    
    def __init__(self):
        self._config = PipelineConfiguration()
    
    def with_data_source(self, folder: str) -> 'PipelineBuilder':
        self._config.data_folder = folder
        return self
    
    def with_clouds(self, mov: str, ref: str) -> 'PipelineBuilder':
        self._config.moving_cloud = mov
        self._config.reference_cloud = ref
        return self
    
    def with_m3c2_params(
        self, 
        normal_scale: float = None,
        search_scale: float = None
    ) -> 'PipelineBuilder':
        if normal_scale:
            self._config.normal_scale = normal_scale
        if search_scale:
            self._config.search_scale = search_scale
        return self
    
    def with_outlier_detection(
        self,
        method: str = "rmse",
        multiplier: float = 3.0
    ) -> 'PipelineBuilder':
        self._config.outlier_method = method
        self._config.outlier_multiplier = multiplier
        return self
    
    def build(self) -> PipelineConfiguration:
        # Validierung
        if not self._config.is_valid():
            raise ValueError("Invalid configuration")
        return self._config