# application/orchestration/pipeline.py
from typing import List, Dict, Any
from domain.commands.base import Command

class Pipeline:
    """Ausführbare Pipeline mit Commands"""
    
    def __init__(self, commands: List[Command]):
        self.commands = commands
    
    def execute(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        context = initial_context.copy()
        
        for command in self.commands:
            if not command.can_execute(context):
                raise RuntimeError(f"Cannot execute {command.__class__.__name__}")
            
            context = command.execute(context)
            
        return context