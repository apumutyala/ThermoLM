"""
Exponential Moving Average (EMA) for ThermoLM JAX

Implements EMA for stable training, following NVIDIA's EDLM approach.

Author: Apuroop Mutyala
Date: April 2026
"""

import jax
import jax.numpy as jnp
import optax
from typing import Any, Dict, Tuple
import jax.tree_util as jtu


class ExponentialMovingAverage:
    """
    Exponential Moving Average for model parameters.
    
    Maintains a shadow copy of parameters updated with EMA.
    Follows NVIDIA's EDLM implementation.
    
    Attributes:
        decay: Decay rate for EMA
        shadow_params: Shadow copy of parameters
    """
    
    def __init__(self, decay: float = 0.9999):
        """
        Initialize EMA.
        
        Args:
            decay: Decay rate for EMA (default: 0.9999)
        """
        self.decay = decay
        self.shadow_params = None
        self.count = 0
    
    def update(self, params: Any) -> None:
        """
        Update shadow parameters with EMA.
        
        Args:
            params: Current model parameters
        """
        if self.shadow_params is None:
            self.shadow_params = jtu.tree_map(lambda x: x.copy(), params)
        else:
            self.shadow_params = jtu.tree_map(
                lambda shadow, param: shadow * self.decay + param * (1 - self.decay),
                self.shadow_params,
                params
            )
        self.count += 1
    
    def copy_to(self, params: Any) -> Any:
        """
        Copy shadow parameters to model parameters.
        
        Args:
            params: Current model parameters
        
        Returns:
            Parameters with shadow values
        """
        return jtu.tree_map(lambda x: x.copy(), self.shadow_params)
    
    def store(self, params: Any) -> None:
        """
        Store current parameters before copying shadow.
        
        Args:
            params: Current model parameters
        """
        self.stored_params = jtu.tree_map(lambda x: x.copy(), params)
    
    def restore(self, params: Any) -> Any:
        """
        Restore stored parameters.
        
        Args:
            params: Current parameters (ignored)
        
        Returns:
            Stored parameters
        """
        return jtu.tree_map(lambda x: x.copy(), self.stored_params)
    
    def state_dict(self) -> dict:
        """
        Get EMA state as dictionary.
        
        Returns:
            Dictionary with shadow_params and count
        """
        return {
            'shadow_params': self.shadow_params,
            'count': self.count
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load EMA state from dictionary.
        
        Args:
            state_dict: Dictionary with shadow_params and count
        """
        self.shadow_params = state_dict['shadow_params']
        self.count = state_dict['count']


def apply_ema(params: Any, ema_params: Any, decay: float) -> Any:
    """
    Apply EMA update to parameters.
    
    Args:
        params: Current parameters
        ema_params: EMA parameters
        decay: Decay rate
    
    Returns:
        Updated EMA parameters
    """
    return jtu.tree_map(
        lambda ema, param: ema * decay + param * (1 - decay),
        ema_params,
        params
    )


# TODO: Add support for bias correction - Implemented below
# TODO: Add support for different decay schedules - Implemented below
# TODO: Test EMA with different decay rates


class EMADecay:
    """Different EMA decay schedules."""
    
    @staticmethod
    def constant(decay: float) -> float:
        """Constant decay rate."""
        return decay
    
    @staticmethod
    def inverse_time(decay: float, step: int) -> float:
        """Inverse time decay (increases over time)."""
        return decay / (1 + step)
    
    @staticmethod
    def exponential(decay: float, step: int, k: float = 0.001) -> float:
        """Exponential decay."""
        return decay * (1 - k) ** step
    
    @staticmethod
    def polynomial(decay: float, step: int, power: float = 2.0) -> float:
        """Polynomial decay."""
        return decay / (1 + step) ** power


def ema_with_bias_correction(
    params: Dict[str, Any],
    ema_params: Dict[str, Any],
    decay: float,
    step: int,
) -> Dict[str, Any]:
    """
    Update EMA parameters with bias correction.
    
    Args:
        params: Current model parameters
        ema_params: EMA parameters
        decay: Decay rate
        step: Current step
    
    Returns:
        updated_ema: Updated EMA parameters with bias correction
    """
    # Compute bias correction factor
    bias_correction = 1 - decay ** (step + 1)
    
    # Update EMA
    def update_ema(param, ema_param):
        return ema_param * decay + param * (1 - decay)
    
    updated_ema = jtu.tree_map(update_ema, params, ema_params)
    
    # Apply bias correction
    def correct_bias(ema_param):
        return ema_param / bias_correction
    
    corrected_ema = jtu.tree_map(correct_bias, updated_ema)
    
    return corrected_ema
