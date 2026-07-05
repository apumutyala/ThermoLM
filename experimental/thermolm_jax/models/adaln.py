"""
Adaptive Layer Normalization (adaLN) for ThermoLM JAX

Implements adaptive layer normalization for timestep conditioning.
Based on NVIDIA's DiT adaLN modulation.

Author: Apuroop Mutyala
Date: April 2026
"""

import flax.linen as nn
import jax.numpy as jnp
from typing import Optional, Tuple


def modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """
    Apply modulation: x * (1 + scale) + shift.
    
    Args:
        x: Input tensor
        shift: Shift parameter
        scale: Scale parameter
    
    Returns:
        Modulated tensor
    """
    return x * (1 + scale) + shift


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization with modulation.
    
    Attributes:
        dim: Dimension of the input
        cond_dim: Dimension of the conditioning (timestep embedding)
    """
    
    dim: int
    cond_dim: int
    
    def setup(self):
        """Initialize adaLN components."""
        self.modulation = nn.Dense(2 * self.dim)
        self.norm = nn.LayerNorm(epsilon=1e-6)
    
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Apply adaptive layer normalization with modulation.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            c: Conditioning tensor of shape (batch, cond_dim)
        
        Returns:
            Tuple of (modulated_x, shift, scale)
        """
        # Normalize
        x_norm = self.norm(x)
        
        # Compute modulation parameters
        shift_scale = self.modulation(c)  # (batch, 2*dim)
        shift, scale = jnp.split(shift_scale, 2, axis=-1)  # (batch, dim) each
        
        # Reshape for broadcasting
        shift = shift[:, None, :]  # (batch, 1, dim)
        scale = scale[:, None, :]  # (batch, 1, dim)
        
        # Apply modulation
        x_modulated = modulate(x_norm, shift, scale)
        
        return x_modulated, shift, scale


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with zero initialization.
    
    Similar to AdaLN but with zero-initialized modulation for stability.
    
    Attributes:
        dim: Dimension of the input
        cond_dim: Dimension of the conditioning
    """
    
    dim: int
    cond_dim: int
    
    def setup(self):
        """Initialize adaLN zero components."""
        # Zero-init via initializers so DiT blocks start as identity at step 0
        self.modulation = nn.Dense(
            2 * self.dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )
        self.norm = nn.LayerNorm(epsilon=1e-6)
    
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Apply adaptive layer normalization with zero initialization.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            c: Conditioning tensor of shape (batch, cond_dim)
        
        Returns:
            Tuple of (modulated_x, shift, scale)
        """
        # Normalize
        x_norm = self.norm(x)
        
        # Compute modulation parameters — c is (batch, cond_dim), no transpose needed
        shift_scale = self.modulation(c)  # (batch, 2*dim)
        shift, scale = jnp.split(shift_scale, 2, axis=-1)

        # Reshape for broadcasting
        shift = shift[:, None, :]
        scale = scale[:, None, :]

        # Apply modulation
        x_modulated = modulate(x_norm, shift, scale)

        return x_modulated, shift, scale


class AdaLNModulation(nn.Module):
    """
    Full adaLN modulation for Diffusion Transformer blocks.
    
    Generates 6 parameters per block:
    - shift_msa, scale_msa, gate_msa
    - shift_mlp, scale_mlp, gate_mlp
    
    Attributes:
        dim: Dimension of the input
        cond_dim: Dimension of the conditioning
    """
    
    dim: int
    cond_dim: int
    
    def setup(self):
        """Initialize adaLN modulation components."""
        # Zero-init so all DiT blocks start as identity at step 0 (DiT paper convention)
        self.modulation = nn.Dense(
            6 * self.dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )
    
    def __call__(self, c: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Generate modulation parameters for Diffusion Transformer block.
        
        Args:
            c: Conditioning tensor of shape (batch, cond_dim)
        
        Returns:
            Tuple of (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        """
        # Compute all modulation parameters — c is (batch, cond_dim), no transpose
        params = self.modulation(c)  # (batch, 6*dim)
        
        # Split into 6 parameters
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = jnp.split(params, 6, axis=-1)
        
        # Reshape for broadcasting
        shift_msa = shift_msa[:, None, :]
        scale_msa = scale_msa[:, None, :]
        gate_msa = gate_msa[:, None, :]
        shift_mlp = shift_mlp[:, None, :]
        scale_mlp = scale_mlp[:, None, :]
        gate_mlp = gate_mlp[:, None, :]
        
        return (shift_msa, scale_msa, gate_msa,
                shift_mlp, scale_mlp, gate_mlp)


# TODO: Add support for class conditioning - Implemented below
# TODO: Add support for classifier-free guidance - Implemented below
# TODO: Test adaLN with different conditioning types


def classifier_free_guidance(
    model_output: jnp.ndarray,
    unconditional_output: jnp.ndarray,
    guidance_scale: float,
) -> jnp.ndarray:
    """
    Apply classifier-free guidance to model output.
    
    Args:
        model_output: Conditional model output
        unconditional_output: Unconditional model output
        guidance_scale: Guidance scale (higher = more guidance)
    
    Returns:
        guided_output: Guided model output
    """
    return unconditional_output + guidance_scale * (model_output - unconditional_output)


class ClassConditionedAdaLN(nn.Module):
    """
    Adaptive Layer Normalization with class conditioning.
    """
    
    d_model: int
    n_classes: int
    dropout: float = 0.1
    
    def setup(self):
        """Initialize class-conditioned adaLN components."""
        # Class embedding
        self.class_embed = nn.Embed(self.n_classes, self.d_model)
        
        # Scale and shift projections
        self.scale_proj = nn.Dense(self.d_model)
        self.shift_proj = nn.Dense(self.d_model)
        
        self.dropout = nn.Dropout(self.dropout)
    
    def __call__(
        self,
        x: jnp.ndarray,
        class_id: jnp.ndarray,
        timestep: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Apply class-conditioned adaptive layer normalization.
        
        Args:
            x: Input embeddings (batch, seq_len, d_model)
            class_id: Class labels (batch,)
            timestep: Optional timestep embedding (batch, d_model)
        
        Returns:
            x: Normalized and conditioned embeddings
        """
        # Get class embedding
        class_emb = self.class_embed(class_id)  # (batch, d_model)
        
        # Combine with timestep if provided
        if timestep is not None:
            conditioning = class_emb + timestep
        else:
            conditioning = class_emb
        
        # Compute scale and shift
        scale = self.scale_proj(conditioning)  # (batch, d_model)
        shift = self.shift_proj(conditioning)  # (batch, d_model)
        
        # Expand for sequence dimension
        scale = scale[:, None, :]  # (batch, 1, d_model)
        shift = shift[:, None, :]  # (batch, 1, d_model)
        
        # Layer normalization
        mean = jnp.mean(x, axis=-1, keepdims=True)
        std = jnp.std(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / (std + 1e-5)
        
        # Apply scale and shift
        x = x_norm * (1 + scale) + shift
        x = self.dropout(x)
        
        return x
