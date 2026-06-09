"""
Timestep Embedding for ThermoLM JAX

Implements sinusoidal timestep embeddings for diffusion models.
Based on NVIDIA's DiT TimestepEmbedder.

Author: Apuroop Mutyala
Date: April 2026
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import math
from typing import Optional


def timestep_embedding(t: jnp.ndarray, dim: int, max_period: int = 10000) -> jnp.ndarray:
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        t: 1-D Tensor of N indices, one per batch element (fractional timesteps allowed)
        dim: Dimension of the output
        max_period: Controls the minimum frequency of the embeddings
    
    Returns:
        (N, D) Tensor of positional embeddings
    """
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half
    )
    args = t[:, None].astype(jnp.float32) * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    
    Uses sinusoidal embeddings followed by an MLP.
    
    Attributes:
        hidden_size: Hidden dimension for the output
        frequency_embedding_size: Dimension of the sinusoidal embeddings
    """
    
    hidden_size: int
    frequency_embedding_size: int = 256
    
    def setup(self):
        """Initialize timestep embedder components."""
        self.mlp = nn.Sequential([
            nn.Dense(self.hidden_size),
            nn.silu,
            nn.Dense(self.hidden_size),
        ])
    
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Embed timesteps.
        
        Args:
            t: Timestep tensor of shape (batch,)
        
        Returns:
            Embedded timesteps of shape (batch, hidden_size)
        """
        # Compute sinusoidal embeddings
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        
        # Pass through MLP
        t_emb = self.mlp(t_freq)
        
        return t_emb


class SinusoidalEmbedding(nn.Module):
    """
    Simple sinusoidal embedding without MLP.
    
    Attributes:
        dim: Dimension of the output
        max_period: Maximum period for frequency computation
    """
    
    dim: int
    max_period: int = 10000
    
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Compute sinusoidal embeddings.
        
        Args:
            t: Timestep tensor of shape (batch,)
        
        Returns:
            Embedded timesteps of shape (batch, dim)
        """
        return timestep_embedding(t, self.dim, self.max_period)


# TODO: Add support for learned positional embeddings - Implemented below
# TODO: Add support for different frequency schedules - Implemented below
# TODO: Test with different embedding dimensions


class LearnedTimestepEmbedding(nn.Module):
    """
    Learned timestep embedding.
    
    Uses learnable embeddings for timesteps instead of sinusoidal encoding.
    """
    
    max_timestep: int
    d_model: int
    dropout: float = 0.1
    
    def setup(self):
        """Initialize learned timestep embedding components."""
        self.embedding = nn.Embed(self.max_timestep, self.d_model)
        self.dropout = nn.Dropout(self.dropout)
    
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Get learned timestep embedding.
        
        Args:
            t: Timestep indices (batch,)
        
        Returns:
            embeddings: Timestep embeddings (batch, d_model)
        """
        # Clip timesteps to valid range
        t_clipped = jnp.clip(t, 0, self.max_timestep - 1).astype(jnp.int32)
        
        # Get embeddings
        embeddings = self.embedding(t_clipped)
        embeddings = self.dropout(embeddings)
        
        return embeddings


def different_frequency_schedules(
    d_model: int,
    schedule_type: str = "sinusoidal",
) -> callable:
    """
    Create timestep embedding with different frequency schedules.
    
    Args:
        d_model: Embedding dimension
        schedule_type: Type of frequency schedule ("sinusoidal", "geometric", "random")
    
    Returns:
        embedding_fn: Function to compute timestep embeddings
    """
    if schedule_type == "sinusoidal":
        def sinusoidal_schedule(t: jnp.ndarray) -> jnp.ndarray:
            """Standard sinusoidal frequency schedule."""
            dim = d_model // 2
            inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim) / dim))
            freqs = t[:, None] * inv_freq[None, :]
            emb = jnp.concatenate([jnp.sin(freqs), jnp.cos(freqs)], axis=-1)
            return emb
        return sinusoidal_schedule
    
    elif schedule_type == "geometric":
        def geometric_schedule(t: jnp.ndarray) -> jnp.ndarray:
            """Geometric frequency schedule."""
            dim = d_model // 2
            inv_freq = 2.0 ** (jnp.arange(0, dim) / dim)
            freqs = t[:, None] / inv_freq[None, :]
            emb = jnp.concatenate([jnp.sin(freqs), jnp.cos(freqs)], axis=-1)
            return emb
        return geometric_schedule
    
    elif schedule_type == "random":
        def random_schedule(t: jnp.ndarray, key: jax.random.PRNGKey = None) -> jnp.ndarray:
            """Random frequency schedule (requires key)."""
            dim = d_model // 2
            if key is None:
                key = jax.random.PRNGKey(42)
            inv_freq = jax.random.uniform(key, (dim,), minval=0.1, maxval=10.0)
            freqs = t[:, None] * inv_freq[None, :]
            emb = jnp.concatenate([jnp.sin(freqs), jnp.cos(freqs)], axis=-1)
            return emb
        return random_schedule
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
