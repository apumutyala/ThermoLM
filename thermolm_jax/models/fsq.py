"""
Finite Scalar Quantization (FSQ) Encoder for ThermoLM JAX.

STATUS: EXPERIMENTAL / NOT VALIDATED. See STATUS.md. This omits FSQ's bounding
nonlinearity (the latent is an unbounded Dense projection before nearest-level
rounding), so codes tend to collapse to the grid boundary.

Implements FSQ from "Finite Scalar Quantization: VQ-VAE Made Simple"
(Mentzer, Minnen, Ballé & Toderici, 2023). Maps continuous embeddings to
discrete codes.

Design Decision: FSQ for discrete encoding
- Rationale: Simpler than VQ-VAE, no codebook collapse, exact reconstruction
- Impact: Better discrete representations for THRML integration
- Trade-off: Limited codebook size compared to learned codebooks
- Downstream: Compatible with THRML's discrete sampling

Author: Apuroop Mutyala
Date: April 14, 2026
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class FSQConfig:
    """Configuration for FSQ encoder."""
    vocab_size: int = 50257  # GPT-2 vocab size
    d_model: int = 512  # Model dimension
    d_latent: int = 64  # Latent dimension
    n_levels: int = 8  # Number of quantization levels per dimension
    max_seq_len: int = 128  # Maximum sequence length


class FSQEncoder(nn.Module):
    """
    Finite Scalar Quantization Encoder.
    
    Maps continuous embeddings to discrete codes using product quantization.
    Each dimension is quantized to one of n_levels values.
    """
    
    config: FSQConfig
    
    @staticmethod
    def _init_levels(n_levels: int) -> jnp.ndarray:
        """
        Initialize quantization levels (static method).
        
        For n_levels, create evenly spaced levels centered at 0.
        Example: n_levels=4 -> [-1.5, -0.5, 0.5, 1.5]
        """
        if n_levels % 2 == 0:
            # Even number of levels: symmetric around 0
            levels = jnp.arange(n_levels) - (n_levels - 1) / 2
        else:
            # Odd number of levels: includes 0
            levels = jnp.arange(n_levels) - (n_levels // 2)
        
        return levels
    
    @property
    def levels(self) -> jnp.ndarray:
        """Get quantization levels."""
        return self._init_levels(self.config.n_levels)
    
    def setup(self):
        """Initialize FSQ encoder components."""
        # Token embedding
        self.token_embed = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.d_model,
        )
        
        # Project to latent space
        self.to_latent = nn.Dense(self.config.d_latent)
        
        # Project back from latent space
        self.from_latent = nn.Dense(self.config.d_model)
    
    def encode(self, tokens: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode tokens to discrete latent codes.
        
        Args:
            tokens: (batch, seq_len) token IDs
        
        Returns:
            codes: (batch, seq_len, d_latent) discrete codes (integers)
            latents: (batch, seq_len, d_latent) continuous latents (before quantization)
        """
        # Embed tokens
        x = self.token_embed(tokens)  # (batch, seq_len, d_model)
        
        # Project to latent space
        latents = self.to_latent(x)  # (batch, seq_len, d_latent)
        
        # Quantize with straight-through estimator
        codes, latents_ste = self.quantize(latents)  # (batch, seq_len, d_latent)
        
        return codes, latents_ste
    
    def decode(self, codes: jnp.ndarray) -> jnp.ndarray:
        """
        Decode discrete codes back to embeddings.
        
        Args:
            codes: (batch, seq_len, d_latent) discrete codes
        
        Returns:
            embeddings: (batch, seq_len, d_model) continuous embeddings
        """
        # Dequantize codes to continuous values
        latents = self.dequantize(codes)  # (batch, seq_len, d_latent)
        
        # Project back to embedding space
        embeddings = self.from_latent(latents)  # (batch, seq_len, d_model)
        
        return embeddings
    
    def quantize(self, latents: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Quantize continuous latents to discrete codes with straight-through estimator.
        
        Args:
            latents: (batch, seq_len, d_latent) continuous values
        
        Returns:
            codes: (batch, seq_len, d_latent) integer codes in range [0, n_levels-1]
            latents_ste: (batch, seq_len, d_latent) straight-through continuous approximation
        """
        # Find nearest level for each dimension
        # levels shape: (n_levels,)
        # latents shape: (batch, seq_len, d_latent)
        
        # Reshape for broadcasting
        latents_expanded = latents[..., None]  # (batch, seq_len, d_latent, 1)
        levels_expanded = self.levels[None, None, None, :]  # (1, 1, 1, n_levels)
        
        # Compute distances to each level
        distances = jnp.abs(latents_expanded - levels_expanded)
        
        # Find nearest level index (this gives indices into levels array)
        code_indices = jnp.argmin(distances, axis=-1)  # (batch, seq_len, d_latent)
        
        # Straight-through estimator: forward uses discrete codes, backward uses continuous latents
        # Get the actual continuous values at the discrete code positions
        code_indices_expanded = code_indices[..., None]  # (batch, seq_len, d_latent, 1)
        latents_at_codes = jnp.take_along_axis(levels_expanded, code_indices_expanded, axis=-1).squeeze(-1)
        
        # STE: forward = latents_at_codes, backward gradient flows through latents
        latents_ste = latents + jax.lax.stop_gradient(latents_at_codes - latents)
        
        # The code_indices are already in range [0, n_levels-1]
        return code_indices.astype(jnp.int32), latents_ste
    
    def dequantize(self, codes: jnp.ndarray) -> jnp.ndarray:
        """
        Dequantize discrete codes to continuous latents.
        
        Args:
            codes: (batch, seq_len, d_latent) integer codes in range [0, n_levels-1]
        
        Returns:
            latents: (batch, seq_len, d_latent) continuous values
        """
        # codes are already indices into levels array
        # Lookup level values
        latents = self.levels[codes]
        
        return latents
    
    def __call__(self, tokens: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Full encode-decode pass (for training).
        
        Args:
            tokens: (batch, seq_len) token IDs
        
        Returns:
            codes: (batch, seq_len, d_latent) discrete codes
            latents: (batch, seq_len, d_latent) continuous latents
            recon_embeddings: (batch, seq_len, d_model) reconstructed embeddings
        """
        codes, latents = self.encode(tokens)
        recon_embeddings = self.decode(codes)
        return codes, latents, recon_embeddings


class FSQDecoder(nn.Module):
    """
    FSQ Decoder for reconstructing from discrete codes.
    
    Can be used separately from encoder for generation.
    """
    
    config: FSQConfig
    
    @property
    def levels(self) -> jnp.ndarray:
        """Get quantization levels."""
        return FSQEncoder._init_levels(self.config.n_levels)
    
    def setup(self):
        """Initialize FSQ decoder components."""
        self.from_latent = nn.Dense(self.config.d_model)
    
    def decode(self, codes: jnp.ndarray) -> jnp.ndarray:
        """
        Decode discrete codes to embeddings.
        
        Args:
            codes: (batch, seq_len, d_latent) discrete codes in range [0, n_levels-1]
        
        Returns:
            embeddings: (batch, seq_len, d_model) continuous embeddings
        """
        # Dequantize
        latents = self.levels[codes]
        
        # Project to embedding space
        embeddings = self.from_latent(latents)
        
        return embeddings
    
    def __call__(self, codes: jnp.ndarray) -> jnp.ndarray:
        """Forward pass."""
        return self.decode(codes)


# TODO: Implement FSQ with learnable levels - Implemented below
# TODO: Add FSQ with different level configurations per dimension - Implemented below
# TODO: Implement FSQ for variable-length sequences - Implemented below


class LearnableFSQEncoder(nn.Module):
    """
    FSQ Encoder with learnable quantization levels.
    """
    
    vocab_size: int
    d_model: int
    d_latent: int
    n_levels: int
    dropout: float = 0.1
    
    def setup(self):
        """Initialize learnable FSQ encoder components."""
        # Token embedding
        self.token_embed = nn.Embed(self.vocab_size, self.d_model)
        
        # Learnable quantization levels
        self.levels = self.param('levels',
            lambda key: jax.random.uniform(key, (self.d_latent, self.n_levels), minval=-1.0, maxval=1.0)
        )
        
        # Projection layers
        self.proj_to_latent = nn.Dense(self.d_latent)
        self.proj_from_latent = nn.Dense(self.d_model)
        
        self.dropout = nn.Dropout(self.dropout)
    
    def quantize(self, latents: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Quantize latents using learnable levels.
        
        Args:
            latents: Continuous latents (batch, seq_len, d_latent)
        
        Returns:
            codes: Quantized codes (batch, seq_len, d_latent)
            quantized_latents: Quantized latents (batch, seq_len, d_latent)
        """
        # Find closest learnable level for each dimension
        batch_size, seq_len, d_latent = latents.shape
        
        # Reshape for processing
        latents_flat = latents.reshape(-1, d_latent)  # (batch*seq_len, d_latent)
        
        # Compute distances to all levels
        levels_expanded = self.levels[None, :, :]  # (1, d_latent, n_levels)
        latents_expanded = latents_flat[:, :, None]  # (batch*seq_len, d_latent, 1)
        
        distances = jnp.abs(levels_expanded - latents_expanded)  # (batch*seq_len, d_latent, n_levels)
        
        # Find closest level
        codes = jnp.argmin(distances, axis=-1)  # (batch*seq_len, d_latent)
        
        # Quantize by selecting closest level
        quantized_flat = jax.vmap(lambda c, lvl: lvl[c], in_axes=(0, None))(codes, self.levels)
        
        # Reshape back
        codes = codes.reshape(batch_size, seq_len, d_latent)
        quantized_latents = quantized_flat.reshape(batch_size, seq_len, d_latent)
        
        return codes, quantized_latents


class PerDimensionFSQ(nn.Module):
    """
    FSQ with different level configurations per dimension.
    """
    
    vocab_size: int
    d_model: int
    d_latent: int
    level_configs: Tuple[int, ...]  # Number of levels per dimension
    dropout: float = 0.1
    
    def setup(self):
        """Initialize per-dimension FSQ components."""
        assert len(self.level_configs) == self.d_latent, "Level configs must match d_latent"
        
        # Token embedding
        self.token_embed = nn.Embed(self.vocab_size, self.d_model)
        
        # Projection layers
        self.proj_to_latent = nn.Dense(self.d_latent)
        self.proj_from_latent = nn.Dense(self.d_model)
        
        self.dropout = nn.Dropout(self.dropout)
    
    def quantize(self, latents: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Quantize latents using per-dimension level configurations.
        
        Args:
            latents: Continuous latents (batch, seq_len, d_latent)
        
        Returns:
            codes: Quantized codes (batch, seq_len, d_latent)
            quantized_latents: Quantized latents (batch, seq_len, d_latent)
        """
        batch_size, seq_len, d_latent = latents.shape
        codes = []
        quantized_latents = []
        
        for i in range(d_latent):
            n_levels = self.level_configs[i]
            latents_dim = latents[:, :, i:i+1]
            
            # Quantize this dimension
            codes_dim = jnp.round((latents_dim + 1) * (n_levels - 1) / 2)
            codes_dim = jnp.clip(codes_dim, 0, n_levels - 1).astype(jnp.int32)
            
            # Dequantize
            quantized_dim = 2 * codes_dim / (n_levels - 1) - 1
            
            codes.append(codes_dim)
            quantized_latents.append(quantized_dim)
        
        codes = jnp.concatenate(codes, axis=-1)
        quantized_latents = jnp.concatenate(quantized_latents, axis=-1)
        
        return codes, quantized_latents


def variable_length_fsq_encode(
    encoder: nn.Module,
    tokens: jnp.ndarray,
    lengths: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Encode variable-length sequences with FSQ.
    
    Args:
        encoder: FSQ encoder
        tokens: Token IDs (batch, max_len)
        lengths: Actual lengths of each sequence (batch,)
    
    Returns:
        codes: Quantized codes (batch, max_len, d_latent)
        latents: Continuous latents (batch, max_len, d_latent)
        mask: Validity mask (batch, max_len)
    """
    batch_size, max_len = tokens.shape
    
    # Create mask
    positions = jnp.arange(max_len)[None, :]
    mask = (positions < lengths[:, None]).astype(jnp.float32)
    
    # Encode
    codes, latents = encoder(tokens)
    
    # Apply mask
    codes = codes * mask[..., None]
    latents = latents * mask[..., None]
    
    return codes, latents, mask
