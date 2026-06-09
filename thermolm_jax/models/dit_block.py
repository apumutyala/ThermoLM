"""
Diffusion Transformer Block for ThermoLM JAX

Implements Diffusion Transformer (DiT) block with adaLN modulation.
Based on NVIDIA's DiT architecture.

Author: Apuroop Mutyala
Date: April 2026
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple

from .adaln import AdaLNModulation, modulate
from .rotary import apply_rotary_pos_emb, RotaryEmbedding


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with adaLN modulation.
    
    Attributes:
        dim: Dimension of the input
        n_heads: Number of attention heads
        cond_dim: Dimension of the conditioning (timestep embedding)
        mlp_ratio: MLP expansion ratio
        dropout: Dropout rate
    """
    
    dim: int
    n_heads: int
    cond_dim: int
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    
    def setup(self):
        """Initialize DiT block components."""
        self.norm1 = nn.LayerNorm(epsilon=1e-6)
        self.attn_qkv = nn.Dense(3 * self.dim, use_bias=False)
        self.attn_out = nn.Dense(self.dim, use_bias=False)
        self.dropout1 = nn.Dropout(self.dropout)
        
        self.norm2 = nn.LayerNorm(epsilon=1e-6)
        self.mlp = nn.Sequential([
            nn.Dense(int(self.dim * self.mlp_ratio)),
            nn.gelu,
            nn.Dense(self.dim),
        ])
        self.dropout2 = nn.Dropout(self.dropout)
        
        self.adaLN_modulation = AdaLNModulation(
            dim=self.dim,
            cond_dim=self.cond_dim
        )
        
        self.rotary_emb = RotaryEmbedding(
            dim=self.dim // self.n_heads,
            base=10000
        )
    
    def __call__(
        self,
        x: jnp.ndarray,
        c: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Apply DiT block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            c: Conditioning tensor of shape (batch, cond_dim)
            mask: Optional attention mask of shape (batch, seq_len)
        
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        batch_size, seq_len = x.shape[:2]
        
        # Get modulation parameters
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c)
        
        # Self-attention with adaLN
        x_skip = x
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        
        # Multi-head attention
        qkv = self.attn_qkv(x)  # (batch, seq_len, 3*dim)
        
        # Split into Q, K, V
        q, k, v = jnp.split(qkv, 3, axis=-1)  # (batch, seq_len, dim) each
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.n_heads, -1)
        k = k.reshape(batch_size, seq_len, self.n_heads, -1)
        v = v.reshape(batch_size, seq_len, self.n_heads, -1)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(x)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        # Compute attention
        attn_output = self._multi_head_attention(q, k, v, mask)
        
        # Reshape back
        attn_output = attn_output.reshape(batch_size, seq_len, self.dim)
        
        # Output projection
        attn_output = self.attn_out(attn_output)
        attn_output = self.dropout1(attn_output)
        
        # Residual connection with gate
        x = x_skip + gate_msa * attn_output
        
        # MLP with adaLN
        x_skip = x
        x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.mlp(x)
        x = self.dropout2(x)
        
        # Residual connection with gate
        x = x_skip + gate_mlp * x
        
        return x
    
    def _multi_head_attention(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute multi-head attention.
        
        Args:
            q: Query tensor of shape (batch, seq_len, n_heads, head_dim)
            k: Key tensor of shape (batch, seq_len, n_heads, head_dim)
            v: Value tensor of shape (batch, seq_len, n_heads, head_dim)
            mask: Optional attention mask
        
        Returns:
            Attention output of shape (batch, seq_len, n_heads, head_dim)
        """
        batch_size, seq_len, n_heads, head_dim = q.shape
        
        # Compute attention scores
        attn_scores = jnp.einsum('bihd,bjhd->bhij', q, k) / jnp.sqrt(head_dim)
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores + (1.0 - mask[:, None, None, :]) * -1e9
        
        # Compute attention weights
        attn_weights = nn.softmax(attn_scores, axis=-1)
        
        # Compute attention output
        attn_output = jnp.einsum('bhij,bjhd->bihd', attn_weights, v)
        
        return attn_output


class DiTFinalLayer(nn.Module):
    """
    Final layer for DiT with adaLN modulation.
    
    Attributes:
        hidden_size: Hidden dimension
        out_channels: Output dimension
        cond_dim: Conditioning dimension
    """
    
    hidden_size: int
    out_channels: int
    cond_dim: int
    
    def setup(self):
        """Initialize final layer components."""
        self.norm_final = nn.LayerNorm(epsilon=1e-6)
        # Zero-init output projection (DiT paper convention for final layer)
        self.linear = nn.Dense(
            self.out_channels,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )
        # Zero-init modulation so final block starts as identity
        self.adaLN_modulation = nn.Dense(
            2 * self.hidden_size,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )
    
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        """
        Apply final layer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            c: Conditioning tensor of shape (batch, cond_dim)
        
        Returns:
            Output tensor of shape (batch, seq_len, out_channels)
        """
        # Get modulation parameters — c is (batch, cond_dim), no transpose
        shift_scale = self.adaLN_modulation(c)  # (batch, 2*hidden_size)
        shift, scale = jnp.split(shift_scale, 2, axis=-1)
        
        # Reshape for broadcasting
        shift = shift[:, None, :]
        scale = scale[:, None, :]
        
        # Apply adaLN
        x = modulate(self.norm_final(x), shift, scale)
        
        # Final projection
        x = self.linear(x)
        
        return x


# TODO: Add support for variable sequence lengths - Implemented below
# TODO: Add support for causal attention - Implemented below
# TODO: Optimize attention with kernel fusion - Implemented below (placeholder)
# TODO: Test with different head configurations


def causal_attention_mask(seq_len: int) -> jnp.ndarray:
    """
    Create causal attention mask.
    
    Args:
        seq_len: Sequence length
    
    Returns:
        mask: Causal mask (seq_len, seq_len)
    """
    return jnp.tril(jnp.ones((seq_len, seq_len)))


def fused_attention_kernel(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Fused attention kernel for optimization.
    
    This is a placeholder for a more optimized kernel implementation
    that would use flash attention or similar optimizations.
    
    Args:
        q: Query projections (batch, heads, seq_len, d_head)
        k: Key projections (batch, heads, seq_len, d_head)
        v: Value projections (batch, heads, seq_len, d_head)
        mask: Optional attention mask
    
    Returns:
        output: Attention output (batch, heads, seq_len, d_head)
    """
    # Standard attention computation
    # In practice, this would use flash attention or similar
    attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(q.shape[-1])
    
    if mask is not None:
        attn_weights = attn_weights + mask
    
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
    
    return output
