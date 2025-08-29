# domain/commands/base.py
"""Command Pattern für Pipeline-Schritte"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PipelineContext:
    """Kontext-Objekt das durch die Pipeline gereicht wird"""
    
    def __init__(self, initial_data: Dict[str, Any] = None):
        self._data = initial_data or {}
        self._history: List[str] = []
        self._errors: List[str] = []
    
    def get(self, key: str, default: Any = None) -> Any:
        """Holt einen Wert aus dem Kontext"""
        return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Setzt einen Wert im Kontext"""
        self._data[key] = value
    
    def update(self, data: Dict[str, Any]) -> None:
        """Aktualisiert mehrere Werte"""
        self._data.update(data)
    
    def has(self, key: str) -> bool:
        """Prüft ob ein Key existiert"""
        return key in self._data
    
    def remove(self, key: str) -> Any:
        """Entfernt und gibt einen Wert zurück"""
        return self._data.pop(key, None)
    
    def add_history(self, entry: str) -> None:
        """Fügt einen History-Eintrag hinzu"""
        self._history.append(entry)
    
    def add_error(self, error: str) -> None:
        """Fügt einen Fehler hinzu"""
        self._errors.append(error)
    
    def get_history(self) -> List[str]:
        """Gibt die History zurück"""
        return self._history.copy()
    
    def get_errors(self) -> List[str]:
        """Gibt die Fehler zurück"""
        return self._errors.copy()
    
    def has_errors(self) -> bool:
        """Prüft ob Fehler aufgetreten sind"""
        return len(self._errors) > 0
    
    def copy(self) -> 'PipelineContext':
        """Erstellt eine Kopie des Kontexts"""
        new_context = PipelineContext(self._data.copy())
        new_context._history = self._history.copy()
        new_context._errors = self._errors.copy()
        return new_context
    
    def to_dict(self) -> Dict[str, Any]:
        """Exportiert den Kontext als Dictionary"""
        return {
            'data': self._data.copy(),
            'history': self._history.copy(),
            'errors': self._errors.copy()
        }


class Command(ABC):
    """Abstrakte Basis für alle Pipeline-Kommandos"""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Führt das Kommando aus.
        
        Args:
            context: Der Pipeline-Kontext
            
        Returns:
            Der aktualisierte Kontext
        """
        pass
    
    @abstractmethod
    def can_execute(self, context: PipelineContext) -> bool:
        """
        Prüft ob das Kommando ausgeführt werden kann.
        
        Args:
            context: Der Pipeline-Kontext
            
        Returns:
            True wenn ausführbar, sonst False
        """
        pass
    
    def validate_preconditions(self, context: PipelineContext) -> List[str]:
        """
        Validiert Vorbedingungen und gibt Fehler zurück.
        
        Args:
            context: Der Pipeline-Kontext
            
        Returns:
            Liste von Fehlermeldungen (leer wenn alles ok)
        """
        return []
    
    def log_execution(self, context: PipelineContext) -> None:
        """Loggt die Ausführung"""
        logger.info(f"Executing command: {self.name}")
        context.add_history(f"Executed: {self.name}")
    
    def handle_error(self, error: Exception, context: PipelineContext) -> None:
        """Behandelt Fehler während der Ausführung"""
        error_msg = f"Error in {self.name}: {str(error)}"
        logger.error(error_msg)
        context.add_error(error_msg)


class CompositeCommand(Command):
    """Zusammengesetztes Kommando das mehrere Sub-Kommandos ausführt"""
    
    def __init__(self, name: str, commands: List[Command]):
        super().__init__(name)
        self.commands = commands
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Führt alle Sub-Kommandos sequenziell aus"""
        self.log_execution(context)
        
        for command in self.commands:
            if not command.can_execute(context):
                error_msg = f"Cannot execute sub-command: {command.name}"
                self.handle_error(RuntimeError(error_msg), context)
                break
            
            try:
                context = command.execute(context)
            except Exception as e:
                self.handle_error(e, context)
                break
        
        return context
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Prüft ob alle Sub-Kommandos ausführbar sind"""
        return all(cmd.can_execute(context) for cmd in self.commands)


class ConditionalCommand(Command):
    """Kommando das nur unter bestimmten Bedingungen ausgeführt wird"""
    
    def __init__(
        self, 
        name: str,
        command: Command,
        condition: callable
    ):
        super().__init__(name)
        self.command = command
        self.condition = condition
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Führt Kommando aus wenn Bedingung erfüllt"""
        if self.condition(context):
            self.log_execution(context)
            return self.command.execute(context)
        else:
            logger.info(f"Skipping {self.name} - condition not met")
            context.add_history(f"Skipped: {self.name} (condition not met)")
            return context
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Prüft Bedingung und Sub-Kommando"""
        if not self.condition(context):
            return True  # Kann ausgeführt werden (wird übersprungen)
        return self.command.can_execute(context)


class ParallelCommand(Command):
    """Führt mehrere Kommandos parallel aus (für zukünftige Erweiterung)"""
    
    def __init__(self, name: str, commands: List[Command]):
        super().__init__(name)
        self.commands = commands
        logger.warning("ParallelCommand is not yet implemented - will run sequentially")
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Führt Kommandos aus (aktuell noch sequenziell)"""
        # TODO: Implementiere echte Parallelisierung mit multiprocessing/threading
        self.log_execution(context)
        
        for command in self.commands:
            if command.can_execute(context):
                try:
                    # In Zukunft: Jedes Kommando bekommt eine Kopie des Kontexts
                    # und Ergebnisse werden am Ende gemerged
                    context = command.execute(context)
                except Exception as e:
                    self.handle_error(e, context)
        
        return context
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Prüft ob mindestens ein Kommando ausführbar ist"""
        return any(cmd.can_execute(context) for cmd in self.commands)