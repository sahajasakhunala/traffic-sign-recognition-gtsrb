import os
from typing import Any, Dict
import yaml
from utils.paths import get_absolute_path

def load_yaml(path: str) -> Dict[str, Any]:
    """Loads a YAML file and returns its dictionary contents.
    
    Resolves relative paths to absolute paths relative to the project root.
    """
    abs_path = get_absolute_path(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"YAML configuration file not found at: {abs_path}")
        
    with open(abs_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    return data or {}

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges multiple configuration dictionaries.
    
    Dictionaries supplied later in the arguments list will overwrite
    matching keys from dictionaries supplied earlier.
    """
    merged = {}
    for config in configs:
        for k, v in config.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = merge_configs(merged[k], v)
            else:
                merged[k] = v
    return merged
