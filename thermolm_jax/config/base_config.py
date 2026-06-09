"""
Base configuration class for ThermoLM JAX.

Provides type-safe, serializable configuration using dataclasses.
Implements DD-ARCH-002.

Author: Apuroop Mutyala
Date: April 14, 2026
"""

from dataclasses import dataclass, asdict, field, fields
from typing import Optional, Dict, Any
import json


@dataclass
class BaseConfig:
    """
    Base configuration class.
    
    Design Decision: Dataclasses for configuration
    - Rationale: Type-safe, serializable, IDE-friendly
    - Impact: Easy to save/load configs, reproduce experiments
    - Trade-off: More verbose than YAML
    - Downstream: Easy experiment reproduction
    
    All configs should inherit from this class to ensure consistent
    serialization and validation.
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create config from dictionary."""
        # Filter out keys that are not fields of the dataclass
        field_names = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BaseConfig':
        """Create config from JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def load(cls, path: str) -> 'BaseConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            return cls.from_json(f.read())
    
    def __str__(self) -> str:
        """String representation."""
        return self.to_json()
