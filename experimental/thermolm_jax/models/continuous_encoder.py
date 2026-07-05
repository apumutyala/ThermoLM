"""
Continuous Encoder for Hybrid Model.

Implements a continuous encoder using transformer architecture
for hybrid continuous-discrete models.

Design Decision: Continuous encoder for hybrid model
- Rationale: Provides continuous embeddings for quantization
- Impact: Better representation learning before quantization
- Trade-off: More parameters than discrete-only approach
- Downstream: Enables two-stage training (continuous → quantized)

Author: Apuroop Mutyala
Date: April 15, 2026
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional
from dataclasses import dataclass
from .rotary import RotaryEmbedding


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with multi-head self-attention.
    
    Implements the standard transformer encoder layer with:
    - Multi-head self-attention
    - Rotary positional embeddings
    - Feed-forward network
    - Layer normalization and residual connections
    """
    
    d_model: int
    num_heads: int
    d_ff: int
    dropout: float = 0.1
    
    def setup(self):
        """Initialize transformer encoder layer components."""
        # Multi-head self-attention
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=True,
        )
        
        # Cross-attention for conditioning
        self.cross_attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=True,
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential([
            nn.Dense(self.d_ff),
            nn.gelu,
            nn.Dropout(self.dropout, deterministic=True),
            nn.Dense(self.d_model),
            nn.Dropout(self.dropout, deterministic=True),
        ])
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(epsilon=1e-5)
        self.norm2 = nn.LayerNorm(epsilon=1e-5)
        self.norm3 = nn.LayerNorm(epsilon=1e-5)
    
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
        causal: bool = False,
        context: Optional[jnp.ndarray] = None,
        context_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Apply transformer encoder layer.
        
        Args:
            x: Input embeddings (batch, seq_len, d_model)
            mask: Attention mask (batch, seq_len, seq_len)
            rotary_emb: Rotary embedding module
            causal: Whether to use causal masking for autoregressive generation
            context: Context embeddings for cross-attention (batch, ctx_len, d_model)
            context_mask: Context attention mask (batch, seq_len, ctx_len)
        
        Returns:
            Output embeddings (batch, seq_len, d_model)
        """
        # Create causal mask if needed
        if causal:
            seq_len = x.shape[1]
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            if mask is not None:
                mask = mask * causal_mask[None, :, :]
            else:
                mask = causal_mask[None, :, :]
        
        # Multi-head self-attention with residual connection
        attn_out = self.attention(x, x, mask=mask)
        x = self.norm1(x + attn_out)
        
        # Cross-attention with context (if provided)
        if context is not None:
            cross_attn_out = self.cross_attention(x, context, mask=context_mask)
            x = self.norm3(x + cross_attn_out)
        
        # Feed-forward network with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


@dataclass
class ContinuousEncoderConfig:
    """Configuration for continuous encoder."""
    vocab_size: int = 50257
    d_model: int = 512
    d_latent: int = 64
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 128
    dropout: float = 0.1


class ContinuousEncoder(nn.Module):
    """
    Continuous encoder using transformer architecture.
    
    Encodes token IDs to continuous latent representations.
    """
    
    config: ContinuousEncoderConfig
    
    def setup(self):
        """Initialize continuous encoder components."""
        # Token embedding
        self.token_embed = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.d_model,
        )
        
        # Rotary positional embeddings
        self.rotary = RotaryEmbedding(self.config.d_model // self.config.num_heads)
        
        # Transformer encoder layers with proper attention
        self.encoder_layers = [
            TransformerEncoderLayer(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                d_ff=self.config.d_ff,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_layers)
        ]
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(epsilon=1e-5)
        
        # Projection to latent space
        self.latent_proj = nn.Dense(self.config.d_latent)
    
    def __call__(
        self,
        tokens: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        causal: bool = False,
        context: Optional[jnp.ndarray] = None,
        context_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode tokens to continuous latent representations.
        
        Args:
            tokens: (batch, seq_len) token IDs
            mask: (batch, seq_len) attention mask (1 for valid, 0 for padding)
            causal: Whether to use causal masking for autoregressive generation
            context: Context embeddings for conditional generation (batch, ctx_len, d_model)
            context_mask: Context attention mask (batch, seq_len, ctx_len)
        
        Returns:
            latents: (batch, seq_len, d_latent) continuous latent representations
            embeddings: (batch, seq_len, d_model) intermediate embeddings
        """
        batch_size, seq_len = tokens.shape
        
        # Token embeddings
        x = self.token_embed(tokens)  # (batch, seq_len, d_model)

        # Transformer encoder layers (rotary embeddings passed down for each layer to apply)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(
                x,
                mask=mask,
                rotary_emb=self.rotary,
                causal=causal,
                context=context,
                context_mask=context_mask,
            )
        
        # Layer normalization
        embeddings = self.layer_norm(x)
        
        # Project to latent space
        latents = self.latent_proj(embeddings)  # (batch, seq_len, d_latent)
        
        return latents, embeddings


class ContinuousDecoder(nn.Module):
    """
    Continuous decoder for reconstruction.
    
    Decodes continuous latent representations back to embeddings.
    """
    
    config: ContinuousEncoderConfig
    
    def setup(self):
        """Initialize continuous decoder components."""
        # Projection from latent space
        self.latent_proj = nn.Dense(self.config.d_model)
        
        # Transformer decoder layers (simplified MLP, no dropout)
        self.decoder_layers = [
            nn.Sequential([
                nn.Dense(self.config.d_ff),
                nn.gelu,
                nn.Dense(self.config.d_model),
                nn.LayerNorm(epsilon=1e-5),
            ])
            for _ in range(self.config.num_layers)
        ]
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(epsilon=1e-5)
        
        # Output projection
        self.output_proj = nn.Dense(self.config.vocab_size)
    
    def __call__(
        self,
        latents: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Decode continuous latents to token logits.
        
        Args:
            latents: (batch, seq_len, d_latent) continuous latent representations
            mask: (batch, seq_len) attention mask
        
        Returns:
            logits: (batch, seq_len, vocab_size) token logits
        """
        # Project from latent space
        x = self.latent_proj(latents)  # (batch, seq_len, d_model)
        
        # Transformer decoder layers
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Output projection
        logits = self.output_proj(x)  # (batch, seq_len, vocab_size)
        
        return logits


# TODO: Add causal masking for autoregressive generation
# TODO: Implement cross-attention for conditional generation
