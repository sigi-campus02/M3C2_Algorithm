# application/factories/service_factory.py
from typing import Dict, Any

class ServiceFactory:
    """Factory für Service-Erstellung mit Dependency Injection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._services = {}
    
    def get_plot_service(self) -> PlotService:
        if 'plot_service' not in self._services:
            self._services['plot_service'] = PlotService(
                repository=self.get_repository(),
                config=self.config.get('plotting', {})
            )
        return self._services['plot_service']
    
    def get_statistics_service(self) -> StatisticsService:
        if 'stats_service' not in self._services:
            self._services['stats_service'] = StatisticsService(
                repository=self.get_repository()
            )
        return self._services['stats_service']
    
    def get_repository(self) -> PointCloudRepository:
        if 'repository' not in self._services:
            self._services['repository'] = FilePointCloudRepository(
                base_path=self.config['data_path']
            )
        return self._services['repository']