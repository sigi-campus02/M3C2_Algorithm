# domain/commands/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class Command(ABC):
    """Basis für alle Pipeline-Kommandos"""
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Führt Kommando aus und gibt aktualisierten Kontext zurück"""
        pass
    
    @abstractmethod
    def can_execute(self, context: Dict[str, Any]) -> bool:
        """Prüft ob Kommando ausgeführt werden kann"""
        pass