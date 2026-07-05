"""
Base data loader interface for ThermoLM JAX.

Provides abstract interface for all data loaders, enabling dataset swapping
without changing training code. Implements DD-ARCH-001.

Author: Apuroop Mutyala
Date: April 14, 2026
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import jax.numpy as jnp


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    All data loaders must implement this interface to ensure compatibility
    with training infrastructure. This enables dataset swapping without
    changing training code (DD-ARCH-001).
    
    Design Decision: Abstract base classes for data loaders
    - Rationale: Enables swapping datasets without changing training code
    - Impact: Future datasets (OpenWebText, C4) can be added easily
    - Trade-off: Slightly more complex than direct implementation
    - Downstream: Future datasets easy to add
    """
    
    @abstractmethod
    def __init__(
        self,
        split: str = 'train',
        max_length: int = 128,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize data loader.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            max_length: Maximum sequence length
            cache_dir: Cache directory for dataset
            **kwargs: Additional dataset-specific parameters
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Get number of examples."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> jnp.ndarray:
        """
        Get a single example.
        
        Returns:
            tokens: (max_length,) token IDs
        """
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        pass
    
    @abstractmethod
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            'num_examples': len(self),
            'max_length': self.max_length,
            'vocab_size': self.get_vocab_size(),
        }
