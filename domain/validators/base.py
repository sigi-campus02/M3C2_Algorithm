# domain/validators/base.py
"""Base Validator mit Chain of Responsibility Pattern"""

from abc import ABC, abstractmethod
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class Validator(ABC):
    """
    Abstrakte Basis für Validatoren mit Chain of Responsibility Pattern.
    
    Validatoren können verkettet werden, um mehrere Validierungen
    nacheinander durchzuführen.
    """
    
    def __init__(self):
        self._next: Optional[Validator] = None
    
    def set_next(self, validator: 'Validator') -> 'Validator':
        """
        Setzt den nächsten Validator in der Kette.
        
        Args:
            validator: Der nächste Validator
            
        Returns:
            Der nächste Validator (für Fluent Interface)
        """
        self._next = validator
        return validator
    
    def validate(self, data: Any) -> bool:
        """
        Führt die Validierung aus und ruft ggf. den nächsten Validator auf.
        
        Args:
            data: Die zu validierende Daten
            
        Returns:
            True wenn alle Validierungen erfolgreich, sonst False
        """
        # Führe eigene Validierung aus
        if not self._validate_impl(data):
            return False
        
        # Rufe nächsten Validator auf wenn vorhanden
        if self._next:
            return self._next.validate(data)
        
        return True
    
    @abstractmethod
    def _validate_impl(self, data: Any) -> bool:
        """
        Implementiert die eigentliche Validierungslogik.
        
        Args:
            data: Die zu validierende Daten
            
        Returns:
            True wenn Validierung erfolgreich, sonst False
        """
        pass
    
    def get_name(self) -> str:
        """Gibt den Namen des Validators zurück"""
        return self.__class__.__name__