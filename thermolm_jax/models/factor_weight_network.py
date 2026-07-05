"""
FactorWeightNetwork: Neural Network that outputs THRML-compatible factor weight tensors.

This is the key conceptual bridge between deep learning and THRML's factor-graph API.
Instead of outputting generic energy scalars, the network explicitly produces factor weight
tensors that can be directly used by THRML's CategoricalEBMFactor and SquareCategoricalEBMFactor.

Phase 2.7: Architectural Redesign - Neural Net as Factor Weight Producer

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional


class TimestepEmbedder(nn.Module):
    """Embed timestep for diffusion models."""
    
    hidden_size: int
    
    def setup(self):
        # Use individual Dense layers instead of Sequential to avoid dimension issues
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(self.hidden_size)
    
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Embed timestep.
        
        Args:
            t: Timestep array, shape (batch,) or (batch, 1)
        
        Returns:
            Embedded timestep, shape (batch, hidden_size)
        """
        if t.ndim == 1:
            t = t[:, None]  # (B, 1)

        # Sinusoidal embedding -> (B, hidden_size). NOTE: previously this used
        # `t[:, None] * emb[None, :]` on an already-(B,1) t, yielding (B,1,H) and
        # a spurious extra batch axis downstream; fixed to keep it (B, H).
        half_dim = self.hidden_size // 2
        scale = jnp.log(10000) / (half_dim - 1)
        freqs = jnp.exp(jnp.arange(half_dim) * -scale)  # (half_dim,)
        ang = t * freqs[None, :]                        # (B, half_dim)
        emb = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)  # (B, hidden)

        emb = nn.silu(self.dense1(emb))
        emb = self.dense2(emb)
        return emb  # (B, hidden_size)


class FactorWeightNetwork(nn.Module):
    """
    Neural network that outputs factor weight tensors for THRML.
    
    Given noisy token sequence x_t, outputs:
    - unary_weights:   (batch, seq_len, n_levels) — affinity of each position for each code
    - pairwise_weights:(batch, seq_len-1, n_levels, n_levels) — adjacent-pair interactions
    
    These are passed directly to CategoricalEBMFactor / SquareCategoricalEBMFactor.
    
    Phase 2.7: Architectural Redesign - Neural Net as Factor Weight Producer
    
    Simplified MLP version for demonstration (avoids transformer dropout/rngs complexity).
    """
    
    vocab_size: int
    hidden_size: int
    n_levels: int
    n_layers: int = 2
    
    def setup(self):
        self.token_embed = nn.Embed(self.vocab_size, self.hidden_size)
        self.time_embed = TimestepEmbedder(self.hidden_size)
        
        # MLP stack with configurable depth
        self.mlp_layers = [nn.Dense(self.hidden_size) for _ in range(self.n_layers)]
        
        # Heads that project to factor weight space
        self.unary_head = nn.Dense(self.n_levels)
        # Pairwise: takes [h_i | h_{i+1}] and produces n_levels x n_levels matrix
        self.pairwise_head = nn.Dense(self.n_levels * self.n_levels)
    
    def __call__(
        self,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute factor weights from noisy tokens.
        
        Args:
            x_t: Noisy token sequence, shape (batch, seq_len)
            t: Timestep, shape (batch,) or (batch, 1)
            mask: Optional attention mask, shape (batch, seq_len)
        
        Returns:
            unary_weights: (batch, seq_len, n_levels) unary factor weights
            pairwise_weights: (batch, seq_len-1, n_levels, n_levels) pairwise factor weights
        """
        # Encode tokens
        h = self.token_embed(x_t)  # (batch, seq, hidden)
        
        # Add timestep embedding
        t_emb = self.time_embed(t)  # (batch, hidden)
        h = h + t_emb[:, None, :]
        
        # MLP stack
        for layer in self.mlp_layers:
            h = nn.silu(layer(h))
        
        # Unary factor weights
        unary_weights = self.unary_head(h)  # (batch, seq, n_levels)
        
        # Pairwise factor weights from concatenated adjacent hidden states
        h_pair = jnp.concatenate([h[:, :-1], h[:, 1:]], axis=-1)  # (batch, seq-1, 2*hidden)
        pw_flat = self.pairwise_head(h_pair)  # (batch, seq-1, n_levels*n_levels)
        pairwise_weights = pw_flat.reshape(
            *pw_flat.shape[:-1], self.n_levels, self.n_levels
        )  # (batch, seq-1, n_levels, n_levels)
        
        return unary_weights, pairwise_weights


def compute_energy_from_weights(
    unary_weights: jnp.ndarray,
    pairwise_weights: jnp.ndarray,
    codes: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute energy E(codes) from factor weights.
    
    E(codes) = -sum(unary_weights[i, codes[i]]) 
               - sum(pairwise_weights[i, codes[i], codes[i+1]])
    
    This is the negative of THRML's energy function (sign convention).
    
    Phase 2.7: Energy computation from factor weights
    
    Args:
        unary_weights:   (batch, seq_len, n_levels) unary factor weights
        pairwise_weights:(batch, seq_len-1, n_levels, n_levels) pairwise factor weights
        codes:           (batch, seq_len) integer codes in [0, n_levels)
    
    Returns:
        energy: (batch,) energy for each sample
    """
    batch, seq_len = codes.shape
    
    # Unary energy: gather weight at each position's code
    unary_energy = -jnp.sum(
        jnp.take_along_axis(unary_weights, codes[..., None], axis=-1).squeeze(-1),
        axis=-1
    )  # (batch,)
    
    # Pairwise energy: gather weight at each adjacent pair's codes
    c_left = codes[:, :-1]  # (batch, seq-1)
    c_right = codes[:, 1:]  # (batch, seq-1)
    
    # Use advanced indexing to gather pairwise weights
    batch_indices = jnp.arange(batch)[:, None]
    pos_indices = jnp.arange(seq_len - 1)[None, :]
    
    pair_energies = -pairwise_weights[
        batch_indices, pos_indices, c_left, c_right
    ]  # (batch, seq-1)
    
    pairwise_energy = jnp.sum(pair_energies, axis=-1)  # (batch,)
    
    return unary_energy + pairwise_energy  # (batch,)
