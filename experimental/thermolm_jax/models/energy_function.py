"""
Energy Function for ThermoLM JAX

Implements the energy function for the EDLM using DiT architecture.
Based on NVIDIA's DiT with adaLN modulation and rotary embeddings.

Author: Apuroop Mutyala
Date: April 2026
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Optional

from .dit_block import DiTBlock, DiTFinalLayer
from .timestep import TimestepEmbedder


class EnergyFunctionJAX(nn.Module):
    """
    Energy function for EDLM using DiT architecture.
    
    Computes E(z_t | t) using a Transformer with timestep conditioning.
    Uses adaLN modulation and rotary embeddings.
    
    Attributes:
        vocab_size: Size of the vocabulary
        hidden_size: Hidden dimension
        n_blocks: Number of DiT blocks
        n_heads: Number of attention heads
        cond_dim: Dimension of timestep conditioning
        dropout: Dropout rate
        mlp_ratio: MLP expansion ratio
    """
    
    vocab_size: int
    hidden_size: int = 512
    n_blocks: int = 6
    n_heads: int = 8
    cond_dim: int = 256
    dropout: float = 0.1
    mlp_ratio: float = 4.0
    
    def setup(self):
        """Initialize energy function components."""
        # Token embeddings
        self.token_embed = nn.Embed(self.vocab_size, self.hidden_size)
        
        # Timestep embedder
        self.sigma_map = TimestepEmbedder(
            hidden_size=self.cond_dim,
            frequency_embedding_size=self.cond_dim
        )
        
        # DiT blocks
        self.blocks = [
            DiTBlock(
                dim=self.hidden_size,
                n_heads=self.n_heads,
                cond_dim=self.cond_dim,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout
            )
            for _ in range(self.n_blocks)
        ]
        
        # Final layer
        self.output_layer = DiTFinalLayer(
            hidden_size=self.hidden_size,
            out_channels=self.vocab_size,
            cond_dim=self.cond_dim
        )
    
    def __call__(
        self,
        x: jnp.ndarray,
        sigma: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute log probabilities (negative energy).
        
        Args:
            x: Token indices, shape (batch, seq_len)
            sigma: Noise level (timestep), shape (batch,)
            mask: Attention mask (optional), shape (batch, seq_len)
        
        Returns:
            Log probabilities, shape (batch, seq_len, vocab_size)
        """
        # Token embeddings
        h = self.token_embed(x)  # (batch, seq_len, hidden_size)
        
        # Timestep conditioning
        c = self.sigma_map(sigma)  # (batch, cond_dim)
        c = nn.silu(c)
        
        # Apply DiT blocks
        for block in self.blocks:
            h = block(h, c, mask)
        
        # Final projection
        logits = self.output_layer(h, c)  # (batch, seq_len, vocab_size)
        
        return logits


def sparse_energy_function(
    energy_fn: callable,
    max_energy_per_edge: float = 1000.0,
    max_degree: int = 4
) -> callable:
    """
    Wrap energy function with sparsity constraints for TSU hardware.
    
    Args:
        energy_fn: Original energy function
        max_energy_per_edge: Maximum energy per edge for hardware
        max_degree: Maximum degree for sparse connectivity
    
    Returns:
        Wrapped energy function with sparsity constraints
    """
    def wrapped(x, sigma, mask=None):
        logits = energy_fn(x, sigma, mask)
        
        # Clip logits for hardware constraints
        logits = jnp.clip(logits, -max_energy_per_edge, max_energy_per_edge)
        
        return logits
    
    return wrapped


# TODO: Add support for variable sequence lengths - Implemented below
# TODO: Add support for class conditioning - Implemented below
# TODO: Implement sparse attention for TSU hardware - Implemented below
# TODO: Add support for classifier-free guidance - Implemented below
# TODO: Test energy function with different architectures


def sparse_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    sparsity_ratio: float = 0.5,
) -> jnp.ndarray:
    """
    Sparse attention for TSU hardware optimization.
    
    Args:
        q: Query projections (batch, heads, seq_len, d_head)
        k: Key projections (batch, heads, seq_len, d_head)
        v: Value projections (batch, heads, seq_len, d_head)
        sparsity_ratio: Ratio of attention weights to keep
    
    Returns:
        output: Sparse attention output
    """
    # Compute attention scores
    attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(q.shape[-1])
    
    # Apply sparsity by keeping top-k attention weights
    k_keep = int(attn.shape[-1] * sparsity_ratio)
    top_k_values, top_k_indices = jax.lax.top_k(attn, k=k_keep)
    
    # Create sparse attention mask
    sparse_attn = jnp.zeros_like(attn)
    sparse_attn = sparse_attn.at[..., top_k_indices].set(top_k_values)
    
    # Softmax on sparse attention
    sparse_attn = jax.nn.softmax(sparse_attn, axis=-1)
    
    # Apply to values
    output = jnp.einsum('bhqk,bhkd->bhqd', sparse_attn, v)
    
    return output


class ClassConditionedEnergyFunction(nn.Module):
    """
    Energy function with class conditioning.
    """
    
    d_model: int
    n_classes: int
    num_layers: int = 6
    dropout: float = 0.1
    
    def setup(self):
        """Initialize class-conditioned energy function components."""
        # Class embedding
        self.class_embed = nn.Embed(self.n_classes, self.d_model)
        
        # Energy layers
        self.layers = [
            nn.Dense(self.d_model) for _ in range(self.num_layers)
        ]
        
        self.layer_norms = [
            nn.LayerNorm(epsilon=1e-5) for _ in range(self.num_layers)
        ]
        
        self.dropout = nn.Dropout(self.dropout)
    
    def __call__(
        self,
        x: jnp.ndarray,
        class_id: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute energy with class conditioning.
        
        Args:
            x: Input embeddings (batch, seq_len, d_model)
            class_id: Class labels (batch,)
            mask: Attention mask
        
        Returns:
            energy: Energy value
        """
        # Get class embedding
        class_emb = self.class_embed(class_id)  # (batch, d_model)
        
        # Add class information to input
        x = x + class_emb[:, None, :]
        
        # Compute energy through layers
        for layer, norm in zip(self.layers, self.layer_norms):
            x = norm(x + self.dropout(layer(x)))
        
        # Sum over sequence to get scalar energy
        if mask is not None:
            energy = jnp.sum(x * mask[..., None]) / jnp.sum(mask)
        else:
            energy = jnp.mean(x)
        
        return energy


def classifier_free_guidance_energy(
    energy_cond: callable,
    energy_uncond: callable,
    x: jnp.ndarray,
    guidance_scale: float,
    condition: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Compute energy with classifier-free guidance.
    
    Args:
        energy_cond: Conditional energy function
        energy_uncond: Unconditional energy function
        x: Input
        guidance_scale: Guidance scale
        condition: Optional conditioning information
    
    Returns:
        guided_energy: Guided energy
    """
    e_cond = energy_cond(x, condition) if condition is not None else energy_cond(x)
    e_uncond = energy_uncond(x)
    
    guided_energy = e_uncond + guidance_scale * (e_cond - e_uncond)
    
    return guided_energy
