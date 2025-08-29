# domain/validators/base.py
from abc import ABC, abstractmethod
from typing import Optional

class Validator(ABC):
    def __init__(self):
        self._next: Optional[Validator] = None
    
    def set_next(self, validator: 'Validator') -> 'Validator':
        self._next = validator
        return validator
    
    def validate(self, data: Any) -> bool:
        if not self._validate_impl(data):
            return False
        
        if self._next:
            return self._next.validate(data)
        
        return True
    
    @abstractmethod
    def _validate_impl(self, data: Any) -> bool:
        pass