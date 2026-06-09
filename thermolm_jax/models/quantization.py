"""
Quantization Layer for Hybrid Model.

Implements various quantization methods for hybrid continuous-discrete models:
- Straight-through quantization
- Vector quantization (VQ-VAE)
- Finite scalar quantization (FSQ)
- Learned quantization

Design Decision: Multiple quantization methods
- Rationale: Enable comparison of quantization approaches
- Impact: Flexible architecture for research
- Trade-off: More complex codebase
- Downstream: Can select best method for specific use case

Author: Apuroop Mutyala
Date: April 15, 2026
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class QuantizationConfig:
    """Configuration for quantization layer."""
    d_latent: int = 64
    n_levels: int = 8
    quantization_type: str = "fsq"  # "fsq", "vq", "learned", "straight_through"
    commitment_cost: float = 0.25
    num_codebooks: int = 1  # For VQ


class StraightThroughQuantization(nn.Module):
    """
    Straight-through quantization with uniform bins.
    
    Simple quantization that passes gradients through straight-through estimator.
    """
    
    config: QuantizationConfig
    
    def setup(self):
        """Initialize straight-through quantization."""
        pass
    
    def __call__(
        self,
        latents: jnp.ndarray,
        train: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """
        Quantize latents using straight-through estimator.
        
        Args:
            latents: (batch, seq_len, d_latent) continuous latents
            train: Whether in training mode
        
        Returns:
            quantized: (batch, seq_len, d_latent) quantized latents
            codes: (batch, seq_len, d_latent) discrete codes
            info: Dictionary with loss and other info
        """
        # Normalize latents to [-1, 1]
        latents_normalized = jnp.tanh(latents)
        
        # Quantize to discrete levels
        levels = jnp.linspace(-1, 1, self.config.n_levels)
        quantized_normalized = jnp.round(
            (latents_normalized + 1) / 2 * (self.config.n_levels - 1)
        ) / (self.config.n_levels - 1) * 2 - 1
        
        # Straight-through estimator
        quantized = latents + jax.lax.stop_gradient(quantized_normalized - latents)
        
        # Convert to integer codes
        codes = jnp.round(
            (quantized_normalized + 1) / 2 * (self.config.n_levels - 1)
        ).astype(jnp.int32)
        
        # Compute quantization loss (commitment loss)
        commitment_loss = jnp.mean((jax.lax.stop_gradient(quantized) - latents) ** 2)
        
        info = {
            'commitment_loss': commitment_loss,
            'quantization_type': 'straight_through',
        }
        
        return quantized, codes, info


class VectorQuantization(nn.Module):
    """
    Vector quantization (VQ-VAE style).
    
    Uses learned codebook for quantization.
    """
    
    config: QuantizationConfig
    
    def setup(self):
        """Initialize VQ codebook."""
        # Learnable codebook
        self.codebook = self.param(
            'codebook',
            nn.initializers.uniform(scale=1.0 / self.config.n_levels),
            (self.config.n_levels, self.config.d_latent),
        )
    
    def __call__(
        self,
        latents: jnp.ndarray,
        train: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """
        Quantize latents using vector quantization.
        
        Args:
            latents: (batch, seq_len, d_latent) continuous latents
            train: Whether in training mode
        
        Returns:
            quantized: (batch, seq_len, d_latent) quantized latents
            codes: (batch, seq_len) codebook indices
            info: Dictionary with loss and other info
        """
        batch_size, seq_len, d_latent = latents.shape
        
        # Flatten
        latents_flat = latents.reshape(-1, d_latent)  # (batch*seq_len, d_latent)
        
        # Compute distances to codebook entries
        # Distance: ||latents - codebook||^2
        distances = jnp.sum(
            latents_flat[:, None, :] ** 2 +
            self.codebook[None, :, :] ** 2 -
            2 * latents_flat[:, None, :] * self.codebook[None, :, :],
            axis=-1,
        )  # (batch*seq_len, n_levels)
        
        # Get nearest codebook entries
        codes_flat = jnp.argmin(distances, axis=-1)  # (batch*seq_len,)
        
        # Quantize
        quantized_flat = self.codebook[codes_flat]  # (batch*seq_len, d_latent)
        
        # Reshape
        codes = codes_flat.reshape(batch_size, seq_len)
        quantized = quantized_flat.reshape(batch_size, seq_len, d_latent)
        
        # Straight-through estimator
        quantized_ste = latents + jax.lax.stop_gradient(quantized - latents)
        
        # Compute losses
        codebook_loss = jnp.mean((jax.lax.stop_gradient(latents) - quantized) ** 2)
        commitment_loss = self.config.commitment_cost * jnp.mean((quantized - jax.lax.stop_gradient(latents)) ** 2)
        
        info = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss,
            'quantization_type': 'vq',
        }
        
        return quantized_ste, codes, info


class FSQQuantization(nn.Module):
    """
    Finite scalar quantization.
    
    Product quantization with different levels per dimension.
    """
    
    config: QuantizationConfig
    
    def setup(self):
        """Initialize FSQ parameters."""
        # Levels for each dimension (product quantization)
        # For simplicity, use same levels for all dimensions
        self.levels = jnp.array([self.config.n_levels] * self.config.d_latent)
        
        # Compute cumulative product for indexing
        self.level_prod = jnp.cumprod(jnp.concatenate([jnp.array([1]), self.levels[:-1]]))
    
    def __call__(
        self,
        latents: jnp.ndarray,
        train: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """
        Quantize latents using FSQ.
        
        Args:
            latents: (batch, seq_len, d_latent) continuous latents
            train: Whether in training mode
        
        Returns:
            quantized: (batch, seq_len, d_latent) quantized latents
            codes: (batch, seq_len, d_latent) discrete codes
            info: Dictionary with loss and other info
        """
        # Normalize latents to [0, 1]
        latents_normalized = (latents + 1) / 2
        
        # Quantize each dimension
        quantized_normalized = jnp.round(
            latents_normalized * (self.config.n_levels - 1)
        ) / (self.config.n_levels - 1)
        
        # Convert to integer codes
        codes = jnp.round(
            latents_normalized * (self.config.n_levels - 1)
        ).astype(jnp.int32)
        
        # Clip to valid range
        codes = jnp.clip(codes, 0, self.config.n_levels - 1)
        
        # Denormalize
        quantized = quantized_normalized * 2 - 1
        
        # Straight-through estimator
        quantized_ste = latents + jax.lax.stop_gradient(quantized - latents)
        
        # Compute commitment loss
        commitment_loss = jnp.mean((jax.lax.stop_gradient(quantized) - latents) ** 2)
        
        info = {
            'commitment_loss': commitment_loss,
            'quantization_type': 'fsq',
        }
        
        return quantized_ste, codes, info


class LearnedQuantization(nn.Module):
    """
    Learned quantization with learnable quantization boundaries.
    
    Learns optimal quantization boundaries during training.
    """
    
    config: QuantizationConfig
    
    def setup(self):
        """Initialize learned quantization parameters."""
        # Learnable quantization boundaries
        self.boundaries = self.param(
            'boundaries',
            nn.initializers.uniform(scale=0.5),
            (self.config.n_levels - 1, self.config.d_latent),
        )
    
    def __call__(
        self,
        latents: jnp.ndarray,
        train: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """
        Quantize latents using learned boundaries.
        
        Args:
            latents: (batch, seq_len, d_latent) continuous latents
            train: Whether in training mode
        
        Returns:
            quantized: (batch, seq_len, d_latent) quantized latents
            codes: (batch, seq_len, d_latent) discrete codes
            info: Dictionary with loss and other info
        """
        batch_size, seq_len, d_latent = latents.shape
        
        # Sort boundaries
        boundaries_sorted = jnp.sort(self.boundaries, axis=0)  # (n_levels-1, d_latent)
        
        # Add -inf and +inf boundaries
        boundaries_full = jnp.concatenate([
            jnp.full((1, d_latent), -jnp.inf),
            boundaries_sorted,
            jnp.full((1, d_latent), jnp.inf),
        ], axis=0)  # (n_levels+1, d_latent)
        
        # Quantize by comparing with boundaries
        codes = jnp.zeros((batch_size, seq_len, d_latent), dtype=jnp.int32)
        
        for i in range(self.config.n_levels):
            # Check if latent is in bin i
            in_bin = (latents >= boundaries_full[i]) & (latents < boundaries_full[i+1])
            codes = jnp.where(in_bin, i, codes)
        
        # Compute quantized values (use bin centers)
        bin_centers = (boundaries_full[:-1] + boundaries_full[1:]) / 2  # (n_levels, d_latent)
        # Simple quantization: use codes to select from bin_centers
        # For each position, use the code to index into bin_centers
        quantized = jnp.zeros_like(latents)
        for i in range(self.config.n_levels):
            quantized = jnp.where(codes == i, bin_centers[i], quantized)
        
        # Straight-through estimator
        quantized_ste = latents + jax.lax.stop_gradient(quantized - latents)
        
        # Compute commitment loss
        commitment_loss = jnp.mean((jax.lax.stop_gradient(quantized) - latents) ** 2)
        
        info = {
            'commitment_loss': commitment_loss,
            'quantization_type': 'learned',
            'boundaries': boundaries_sorted,
        }
        
        return quantized_ste, codes, info


class QuantizationLayer(nn.Module):
    """
    Unified quantization layer supporting multiple methods.
    """
    
    config: QuantizationConfig
    
    def setup(self):
        """Initialize quantization layer."""
        if self.config.quantization_type == "straight_through":
            self.quantization = StraightThroughQuantization(self.config)
        elif self.config.quantization_type == "vq":
            self.quantization = VectorQuantization(self.config)
        elif self.config.quantization_type == "fsq":
            self.quantization = FSQQuantization(self.config)
        elif self.config.quantization_type == "learned":
            self.quantization = LearnedQuantization(self.config)
        else:
            raise ValueError(f"Unknown quantization type: {self.config.quantization_type}")
    
    def __call__(
        self,
        latents: jnp.ndarray,
        train: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """
        Quantize latents using configured method.
        
        Args:
            latents: (batch, seq_len, d_latent) continuous latents
            train: Whether in training mode
        
        Returns:
            quantized: (batch, seq_len, d_latent) quantized latents
            codes: (batch, seq_len, d_latent) discrete codes
            info: Dictionary with loss and other info
        """
        return self.quantization(latents, train=train)


# TODO: Add entropy regularization for better quantization - Implemented below
# TODO: Implement codebook reset for VQ to prevent codebook collapse - Implemented below
# TODO: Add different level configurations per dimension for FSQ - Implemented in fsq.py


def entropy_regularization(codes: jnp.ndarray, num_codes: int, beta: float = 0.1) -> jnp.ndarray:
    """
    Compute entropy regularization to encourage uniform code usage.
    
    Args:
        codes: Quantized codes (batch, seq_len, d_latent)
        num_codes: Total number of possible codes
        beta: Regularization strength
    
    Returns:
        entropy_loss: Entropy regularization loss
    """
    # Flatten codes
    codes_flat = codes.reshape(-1)
    
    # Count code usage
    code_counts = jnp.bincount(codes_flat, length=num_codes)
    
    # Normalize to probabilities
    code_probs = code_counts / jnp.sum(code_counts)
    
    # Compute entropy
    entropy = -jnp.sum(code_probs * jnp.log(code_probs + 1e-10))
    
    # Regularization: encourage high entropy (uniform usage)
    max_entropy = jnp.log(num_codes)
    entropy_loss = beta * (max_entropy - entropy)
    
    return entropy_loss


def codebook_reset(
    codebook: jnp.ndarray,
    usage_counts: jnp.ndarray,
    threshold: float = 0.01,
    reset_value: float = 0.0,
) -> jnp.ndarray:
    """
    Reset unused codebook entries to prevent codebook collapse.
    
    Args:
        codebook: Codebook embeddings (num_codes, d_model)
        usage_counts: Usage count for each code (num_codes,)
        threshold: Usage threshold below which codes are reset
        reset_value: Value to reset codes to
    
    Returns:
        updated_codebook: Updated codebook
    """
    total_usage = jnp.sum(usage_counts)
    if total_usage > 0:
        usage_freq = usage_counts / total_usage
    else:
        usage_freq = jnp.zeros_like(usage_counts)
    
    # Identify unused codes
    unused_mask = usage_freq < threshold
    
    # Reset unused codes
    updated_codebook = jnp.where(
        unused_mask[:, None],
        jnp.ones_like(codebook) * reset_value,
        codebook
    )
    
    return updated_codebook


def adaptive_level_configs(
    d_latent: int,
    target_codebook_size: int,
) -> Tuple[int, ...]:
    """
    Compute level configurations to achieve target codebook size.
    
    Args:
        d_latent: Number of latent dimensions
        target_codebook_size: Target total codebook size
    
    Returns:
        level_configs: Level configuration per dimension
    """
    # Find configuration that approximates target size
    # For simplicity, use uniform levels
    levels_per_dim = int(round(target_codebook_size ** (1 / d_latent)))
    
    level_configs = tuple([levels_per_dim] * d_latent)
    
    return level_configs
