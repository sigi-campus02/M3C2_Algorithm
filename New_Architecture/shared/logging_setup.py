# shared/logging_setup.py
"""Zentrales Logging-Setup für die Anwendung"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Dict, Any
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Formatter mit Farben für Console-Output"""
    
    # ANSI Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Füge Farbe hinzu wenn auf Console
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        return super().format(record)


def setup_logging(
    config: Optional[Dict[str, Any]] = None,
    log_file: Optional[str] = None,
    level: Optional[str] = None,
    console: bool = True,
    file: bool = True
) -> None:
    """
    Konfiguriert Application-wide Logging.
    
    Args:
        config: Konfigurations-Dictionary mit logging-Sektion
        log_file: Pfad zur Log-Datei (überschreibt config)
        level: Log-Level (überschreibt config)
        console: Ob auf Console geloggt werden soll
        file: Ob in Datei geloggt werden soll
    """
    
    # Default-Werte
    if config and 'logging' in config:
        log_config = config['logging']
    else:
        log_config = {}
    
    # Überschreibe mit expliziten Parametern
    if log_file is None:
        log_file = log_config.get('file', 'logs/orchestration.log')
    if level is None:
        level = log_config.get('level', 'INFO')
    
    # Konvertiere Level-String zu logging-Konstante
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Format für Log-Nachrichten
    log_format = log_config.get(
        'format',
        '%(asctime)s [%(levelname)-8s] %(name)s - %(message)s'
    )
    date_format = log_config.get('date_format', '%Y-%m-%d %H:%M:%S')
    
    # Root-Logger konfigurieren
    root = logging.getLogger()
    root.setLevel(numeric_level)
    
    # Entferne bestehende Handler
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    
    # Console Handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        # Verwende ColoredFormatter für Console
        console_formatter = ColoredFormatter(log_format, datefmt=date_format)
        console_handler.setFormatter(console_formatter)
        
        root.addHandler(console_handler)
    
    # File Handler
    if file and log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Wähle Handler-Typ basierend auf Konfiguration
        rotation_type = log_config.get('rotation', 'size')
        
        if rotation_type == 'size':
            # Größenbasierte Rotation
            max_bytes = log_config.get('max_bytes', 10 * 1024 * 1024)  # 10MB default
            backup_count = log_config.get('backup_count', 5)
            
            file_handler = RotatingFileHandler(
                str(log_path),
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        elif rotation_type == 'time':
            # Zeitbasierte Rotation
            when = log_config.get('when', 'midnight')  # täglich um Mitternacht
            interval = log_config.get('interval', 1)
            backup_count = log_config.get('backup_count', 30)
            
            file_handler = TimedRotatingFileHandler(
                str(log_path),
                when=when,
                interval=interval,
                backupCount=backup_count
            )
        else:
            # Einfacher File-Handler ohne Rotation
            file_handler = logging.FileHandler(str(log_path))
        
        file_handler.setLevel(numeric_level)
        
        # Standard-Formatter für Datei
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        
        root.addHandler(file_handler)
    
    # Spezielle Logger-Konfiguration
    configure_module_loggers(log_config.get('modules', {}))
    
    # Log Startup-Nachricht
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info(f"Logging initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log level: {level}")
    if file and log_file:
        logger.info(f"Log file: {log_path.absolute()}")
    logger.info("=" * 60)


def configure_module_loggers(module_config: Dict[str, str]) -> None:
    """
    Konfiguriert spezielle Log-Level für bestimmte Module.
    
    Args:
        module_config: Dict mit Modul-Namen als Keys und Log-Level als Values
    """
    for module_name, level in module_config.items():
        logger = logging.getLogger(module_name)
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        logging.getLogger(__name__).debug(
            f"Set log level for {module_name} to {level}"
        )


def get_logger(name: str) -> logging.Logger:
    """
    Convenience-Funktion zum Abrufen eines Loggers.
    
    Args:
        name: Name des Loggers (normalerweise __name__)
        
    Returns:
        Logger-Instanz
    """
    return logging.getLogger(name)


class LogContext:
    """Context Manager für temporäre Log-Level-Änderungen"""
    
    def __init__(self, logger_name: str, level: str):
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
        self.new_level = getattr(logging, level.upper(), logging.INFO)
    
    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)


def log_execution_time(func):
    """Decorator zum Loggen der Ausführungszeit einer Funktion"""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.perf_counter()
        
        logger.debug(f"Starting {func.__name__}")
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            logger.debug(f"Completed {func.__name__} in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"Failed {func.__name__} after {elapsed:.3f}s: {e}")
            raise
    
    return wrapper


def log_exceptions(logger_name: Optional[str] = None):
    """Decorator zum Loggen von Exceptions"""
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name or func.__module__)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Exception in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator