# New_Architecture/application/factories/extended_service_factory.py
"""Erweiterte Service Factory mit Enhanced Visualization Service"""

from application.factories.service_factory import ServiceFactory
from application.services.enhanced_visualization_service import EnhancedVisualizationService


class ExtendedServiceFactory(ServiceFactory):
    """Erweiterte Service Factory mit zusätzlichen Services"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self._enhanced_viz_service = None
    
    def get_enhanced_visualization_service(self) -> EnhancedVisualizationService:
        """Gibt Enhanced Visualization Service zurück (Singleton)"""
        if self._enhanced_viz_service is None:
            self._enhanced_viz_service = EnhancedVisualizationService()
        return self._enhanced_viz_service
    
    def get_visualization_service(self):
        """Override: Nutze Enhanced Visualization Service"""
        return self.get_enhanced_visualization_service()