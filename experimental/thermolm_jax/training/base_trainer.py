"""
Base trainer interface for ThermoLM JAX.

Provides abstract interface for all trainers, enabling different training
strategies to share infrastructure. Implements DD-ARCH-003.

Author: Apuroop Mutyala
Date: April 14, 2026
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import jax
import jax.numpy as jnp
from ..config import BaseConfig


class BaseTrainer(ABC):
    """
    Abstract base class for trainers.
    
    All trainers must implement this interface to ensure compatibility
    with training infrastructure. This enables different training
    strategies (discrete, hybrid, continuous) to share infrastructure
    (DD-ARCH-003).
    
    Design Decision: Abstract trainer interface
    - Rationale: Different strategies may need different training loops
    - Impact: Discrete, hybrid, continuous can share infrastructure
    - Trade-off: More abstraction than simple training script
    - Downstream: Future training strategies easy to add
    """
    
    def __init__(self, config: BaseConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
    
    @abstractmethod
    def train_step(
        self,
        params: Dict[str, Any],
        opt_state: Any,
        batch: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> tuple:
        """
        Single training step.
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            batch: Training batch
            key: PRNG key
        
        Returns:
            (params, opt_state, loss, metrics, key)
        """
        pass
    
    @abstractmethod
    def validation_step(
        self,
        params: Dict[str, Any],
        batch: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """
        Single validation step.
        
        Args:
            params: Model parameters
            batch: Validation batch
        
        Returns:
            metrics: Dictionary of validation metrics
        """
        pass
    
    @abstractmethod
    def train(
        self,
        train_data: jnp.ndarray,
        valid_data: Optional[jnp.ndarray] = None,
        num_epochs: int = 10,
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            train_data: Training data
            valid_data: Validation data (optional)
            num_epochs: Number of training epochs
        
        Returns:
            trained_params: Trained model parameters
            history: Training history (losses, metrics, etc.)
        """
        pass
    
    def get_checkpoint(self, params: Dict[str, Any], step: int) -> Dict[str, Any]:
        """
        Get checkpoint dictionary.
        
        Args:
            params: Model parameters
            step: Training step
        
        Returns:
            checkpoint: Checkpoint dictionary
        """
        return {
            'params': params,
            'step': step,
            'config': self.config.to_dict(),
        }
