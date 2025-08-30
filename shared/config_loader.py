# shared/config_loader.py
"""Configuration Loader für verschiedene Formate"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import configparser

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Lädt Konfigurationen aus verschiedenen Quellen"""
    
    @staticmethod
    def load(config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Lädt Konfiguration aus Datei oder verwendet Defaults.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            
        Returns:
            Dictionary mit Konfigurationswerten
        """
        config = ConfigLoader.get_defaults()
        
        if config_path:
            file_config = ConfigLoader.load_from_file(config_path)
            config = ConfigLoader.merge_configs(config, file_config)
        
        # Lade Umgebungsvariablen
        env_config = ConfigLoader.load_from_env()
        config = ConfigLoader.merge_configs(config, env_config)
        
        logger.info(f"Configuration loaded: {list(config.keys())}")
        return config
    
    @staticmethod
    def load_from_file(path: str) -> Dict[str, Any]:
        """Lädt Konfiguration aus Datei"""
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            return ConfigLoader._load_json(file_path)
        elif suffix in ['.yml', '.yaml']:
            return ConfigLoader._load_yaml(file_path)
        elif suffix in ['.ini', '.cfg']:
            return ConfigLoader._load_ini(file_path)
        else:
            raise ValueError(f"Unsupported configuration format: {suffix}")
    
    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        """Lädt JSON-Konfiguration"""
        with open(path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded JSON config from {path}")
        return config
    
    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Lädt YAML-Konfiguration"""
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded YAML config from {path}")
            return config or {}
        except ImportError:
            logger.warning("PyYAML not installed. Cannot load YAML config.")
            return {}
    
    @staticmethod
    def _load_ini(path: Path) -> Dict[str, Any]:
        """Lädt INI-Konfiguration"""
        parser = configparser.ConfigParser()
        parser.read(path)
        
        config = {}
        for section in parser.sections():
            config[section] = {}
            for key, value in parser.items(section):
                # Versuche Werte zu parsen
                try:
                    # Boolean
                    if value.lower() in ['true', 'false']:
                        config[section][key] = value.lower() == 'true'
                    # Integer
                    elif value.isdigit():
                        config[section][key] = int(value)
                    # Float
                    elif '.' in value:
                        try:
                            config[section][key] = float(value)
                        except ValueError:
                            config[section][key] = value
                    else:
                        config[section][key] = value
                except:
                    config[section][key] = value
        
        logger.info(f"Loaded INI config from {path}")
        return config
    
    @staticmethod
    def load_from_env() -> Dict[str, Any]:
        """Lädt Konfiguration aus Umgebungsvariablen"""
        import os
        
        config = {}
        prefix = "M3C2_"  # Prefix für relevante Umgebungsvariablen
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Entferne Prefix und konvertiere zu lowercase
                config_key = key[len(prefix):].lower()
                
                # Versuche Wert zu parsen
                if value.lower() in ['true', 'false']:
                    config[config_key] = value.lower() == 'true'
                elif value.isdigit():
                    config[config_key] = int(value)
                else:
                    try:
                        config[config_key] = float(value)
                    except ValueError:
                        config[config_key] = value
        
        if config:
            logger.info(f"Loaded {len(config)} values from environment variables")
        
        return config
    
    @staticmethod
    def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mergt zwei Konfigurationen (override überschreibt base).
        
        Args:
            base: Basis-Konfiguration
            override: Überschreibende Konfiguration
            
        Returns:
            Gemergte Konfiguration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Rekursiv mergen für verschachtelte Dictionaries
                result[key] = ConfigLoader.merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def get_defaults() -> Dict[str, Any]:
        """Gibt Default-Konfiguration zurück"""
        return {
            # Daten-Pfade
            'data_path': 'data',
            'output_path': 'outputs',
            
            # M3C2 Parameter
            'm3c2': {
                'normal_scale': None,  # Auto-detect wenn None
                'search_scale': None,  # Auto-detect wenn None
                'use_existing_params': False,
            },
            
            # Ausreißer-Erkennung
            'outlier_detection': {
                'method': 'rmse',
                'multiplier': 3.0,
            },
            
            # Verarbeitung
            'processing': {
                'mov_as_corepoints': True,
                'use_subsampled_corepoints': 1,
                'sample_size': 10000,
                'only_stats': False,
                'stats_type': 'distance',
                'version': 'python',
            },
            
            # Output
            'output': {
                'format': 'excel',
                'project_name': 'default_project',
            },
            
            # Plotting
            'plotting': {
                'bins': 256,
                'dpi': 100,
                'figure_size': (10, 6),
            },
            
            # Logging
            'logging': {
                'level': 'INFO',
                'file': 'orchestration.log',
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            }
        }
    
    @staticmethod
    def save(config: Dict[str, Any], path: str) -> None:
        """Speichert Konfiguration in Datei"""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
        elif suffix in ['.yml', '.yaml']:
            try:
                with open(file_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            except ImportError:
                logger.error("PyYAML not installed. Cannot save YAML config.")
                raise
        else:
            raise ValueError(f"Unsupported format for saving: {suffix}")
        
        logger.info(f"Saved configuration to {file_path}")
    
    @staticmethod
    def validate(config: Dict[str, Any]) -> bool:
        """
        Validiert eine Konfiguration.
        
        Returns:
            True wenn gültig, sonst False
        """
        required_keys = ['data_path', 'output_path']
        
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required configuration key: {key}")
                return False
        
        # Prüfe ob Pfade existieren
        data_path = Path(config['data_path'])
        if not data_path.exists():
            logger.warning(f"Data path does not exist: {data_path}")
        
        # Validiere Wertebereiche
        if 'outlier_detection' in config:
            multiplier = config['outlier_detection'].get('multiplier', 3.0)
            if multiplier <= 0:
                logger.error(f"Invalid outlier multiplier: {multiplier}")
                return False
        
        return True