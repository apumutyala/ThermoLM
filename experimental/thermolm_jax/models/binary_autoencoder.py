"""
Binary Autoencoder for Language-to-DTM Mapping

Implements autoencoder to map language tokens to binary spins for DTM.
Follows Extropic.pdf Appendix I three-stage training approach.

Design Decision: Binary Autoencoder with STE
- Rationale: Maps continuous language to binary DTM inputs
- Impact: Enables DTM for language modeling
- Trade-off: Binary quantization loses some information
- Downstream: Required for language DTM

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import equinox as eqx
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class BinaryAutoencoderConfig:
    """Configuration for binary autoencoder."""
    vocab_size: int = 50257  # GPT-2 vocabulary size
    d_model: int = 512  # Model dimension
    d_latent: int = 64  # Number of binary spins per token
    max_seq_len: int = 128  # Maximum sequence length
    num_layers: int = 6  # Number of transformer layers
    num_heads: int = 8  # Number of attention heads
    dropout: float = 0.1  # Dropout rate


class BinaryQuantizer(eqx.Module):
    """
    Binary quantizer with straight-through estimator.
    
    Maps continuous features to {-1, +1} binary spins using:
    1. Tanh activation to bound to [-1, 1]
    2. Sign function for binarization
    3. Straight-through estimator for gradients
    """
    
    projection: eqx.nn.Linear
    
    def __init__(self, d_model: int, d_latent: int, key: jax.random.PRNGKey):
        self.projection = eqx.nn.Linear(d_model, d_latent, key=key)
    
    def __call__(self, h: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Quantize continuous features to binary spins.
        
        Args:
            h: Continuous features, shape (..., d_model)
        
        Returns:
            binary: Binary spins {-1, +1}, shape (..., d_latent)
            h_q: Quantized features for gradient flow, shape (..., d_latent)
        """
        # Project to latent space
        h_proj = jnp.tanh(self.projection(h))
        
        # Binarize to {-1, +1}
        binary = jnp.sign(h_proj)
        binary = jnp.where(binary == 0, jnp.ones_like(binary), binary)
        
        # Straight-through estimator
        h_q = h_proj + (binary - h_proj).detach()
        
        return binary, h_q


class BinaryEncoder(eqx.Module):
    """
    Transformer encoder with binary quantization.
    
    Maps language tokens to binary spin sequences.
    """
    
    config: BinaryAutoencoderConfig
    token_embed: eqx.nn.Embedding
    pos_embed: eqx.nn.Embedding
    quantizer: BinaryQuantizer
    transformer_layers: list
    
    def __init__(self, config: BinaryAutoencoderConfig, key: jax.random.PRNGKey):
        self.config = config
        
        key_embed, key_pos, key_quant, *transformer_keys = jax.random.split(key, 3 + config.num_layers)
        
        # Token embedding
        self.token_embed = eqx.nn.Embedding(
            config.vocab_size,
            config.d_model,
            key=key_embed
        )
        
        # Positional embedding
        self.pos_embed = eqx.nn.Embedding(
            config.max_seq_len,
            config.d_model,
            key=key_pos
        )
        
        # Binary quantizer
        self.quantizer = BinaryQuantizer(config.d_model, config.d_latent, key_quant)
        
        # Transformer layers (simplified for now)
        self.transformer_layers = []
        for i in range(config.num_layers):
            layer_key = transformer_keys[i]
            # Simple MLP for now (can be replaced with full transformer)
            layer = eqx.nn.Sequential([
                eqx.nn.Linear(config.d_model, config.d_model * 4, key=layer_key),
                jax.nn.relu,
                eqx.nn.Linear(config.d_model * 4, config.d_model, key=layer_key),
            ])
            self.transformer_layers.append(layer)
    
    def __call__(
        self,
        tokens: jnp.ndarray,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode tokens to binary spins.
        
        Args:
            tokens: Token IDs, shape (batch_size, seq_len)
            key: Random key (for potential stochasticity)
        
        Returns:
            binary: Binary spins, shape (batch_size, seq_len, d_latent)
            commitment_loss: Commitment loss for training
        """
        batch_size, seq_len = tokens.shape
        
        # Embed tokens
        h = self.token_embed(tokens)
        
        # Add positional encoding
        positions = jnp.arange(seq_len)
        pos_emb = self.pos_embed(positions)
        h = h + pos_emb[None, :, :]
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            h = layer(h)
        
        # Quantize to binary
        binary, h_q = self.quantizer(h)
        
        # Commitment loss (encourage projection to match quantization)
        h_proj = jnp.tanh(self.quantizer.projection(h))
        commitment_loss = jnp.mean(jnp.square(h_proj - h_q.detach()))
        
        return binary, commitment_loss


class BinaryDecoder(eqx.Module):
    """
    Transformer decoder for binary spins to language.
    
    Maps binary spin sequences back to token logits.
    """
    
    config: BinaryAutoencoderConfig
    binary_embed: eqx.nn.Linear
    pos_embed: eqx.nn.Embedding
    transformer_layers: list
    output_proj: eqx.nn.Linear
    
    def __init__(self, config: BinaryAutoencoderConfig, key: jax.random.PRNGKey):
        self.config = config
        
        key_binary, key_pos, key_output, *transformer_keys = jax.random.split(key, 3 + config.num_layers)
        
        # Binary embedding
        self.binary_embed = eqx.nn.Linear(config.d_latent, config.d_model, key=key_binary)
        
        # Positional embedding
        self.pos_embed = eqx.nn.Embedding(
            config.max_seq_len,
            config.d_model,
            key=key_pos
        )
        
        # Transformer layers
        self.transformer_layers = []
        for i in range(config.num_layers):
            layer_key = transformer_keys[i]
            layer = eqx.nn.Sequential([
                eqx.nn.Linear(config.d_model, config.d_model * 4, key=layer_key),
                jax.nn.relu,
                eqx.nn.Linear(config.d_model * 4, config.d_model, key=layer_key),
            ])
            self.transformer_layers.append(layer)
        
        # Output projection
        self.output_proj = eqx.nn.Linear(config.d_model, config.vocab_size, key=key_output)
    
    def __call__(self, binary: jnp.ndarray) -> jnp.ndarray:
        """
        Decode binary spins to token logits.
        
        Args:
            binary: Binary spins, shape (batch_size, seq_len, d_latent)
        
        Returns:
            logits: Token logits, shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len, d_latent = binary.shape
        
        # Embed binary spins
        h = self.binary_embed(binary.astype(jnp.float32))
        
        # Add positional encoding
        positions = jnp.arange(seq_len)
        pos_emb = self.pos_embed(positions)
        h = h + pos_emb[None, :, :]
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            h = layer(h)
        
        # Project to vocabulary
        logits = self.output_proj(h)
        
        return logits


class BinaryAutoencoder(eqx.Module):
    """
    Complete binary autoencoder for language-to-DTM mapping.
    
    Implements the three-stage training approach from Extropic.pdf:
    1. Train autoencoder (reconstruction loss)
    2. Train DTM on binary embeddings (contrastive divergence)
    3. GAN fine-tune decoder
    """
    
    encoder: BinaryEncoder
    decoder: BinaryDecoder
    
    def __init__(self, config: BinaryAutoencoderConfig, key: jax.random.PRNGKey):
        key_enc, key_dec = jax.random.split(key)
        self.encoder = BinaryEncoder(config, key_enc)
        self.decoder = BinaryDecoder(config, key_dec)
    
    def encode(
        self,
        tokens: jnp.ndarray,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode tokens to binary spins.
        
        Args:
            tokens: Token IDs
            key: Random key
        
        Returns:
            binary: Binary spins
            commitment_loss: Commitment loss
        """
        return self.encoder(tokens, key)
    
    def decode(self, binary: jnp.ndarray) -> jnp.ndarray:
        """
        Decode binary spins to token logits.
        
        Args:
            binary: Binary spins
        
        Returns:
            logits: Token logits
        """
        return self.decoder(binary)
    
    def reconstruction_loss(
        self,
        tokens: jnp.ndarray,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute reconstruction loss.
        
        Args:
            tokens: Token IDs
            key: Random key
        
        Returns:
            total_loss: Total loss (reconstruction + commitment)
            recon_loss: Cross-entropy reconstruction loss
            commitment_loss: Quantization commitment loss
        """
        # Encode
        binary, commitment_loss = self.encode(tokens, key)
        
        # Decode
        logits = self.decode(binary)
        
        # Reconstruction loss (cross-entropy)
        recon_loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, tokens)
        )
        
        # Total loss
        total_loss = recon_loss + 0.25 * commitment_loss
        
        return total_loss, recon_loss, commitment_loss


def test_binary_autoencoder():
    """Test binary autoencoder implementation."""
    print("Testing BinaryAutoencoder...")
    
    config = BinaryAutoencoderConfig(
        vocab_size=1000,
        d_model=128,
        d_latent=16,
        max_seq_len=32,
        num_layers=2
    )
    
    key = jax.random.PRNGKey(0)
    autoencoder = BinaryAutoencoder(config, key)
    
    # Test encoding
    tokens = jax.random.randint(key, (4, 16), minval=0, maxval=1000)
    binary, commitment_loss = autoencoder.encode(tokens)
    
    assert binary.shape == (4, 16, 16), f"Expected (4, 16, 16), got {binary.shape}"
    print(f"Binary shape: {binary.shape}")
    print(f"Binary values: {binary[0, 0, :5]}...")
    print(f"Commitment loss: {commitment_loss}")
    
    # Test decoding
    logits = autoencoder.decode(binary)
    assert logits.shape == (4, 16, 1000), f"Expected (4, 16, 1000), got {logits.shape}"
    print(f"Logits shape: {logits.shape}")
    
    # Test reconstruction loss
    total_loss, recon_loss, commit_loss = autoencoder.reconstruction_loss(tokens)
    print(f"Total loss: {total_loss}")
    print(f"Reconstruction loss: {recon_loss}")
    print(f"Commitment loss: {commit_loss}")
    
    print("[SUCCESS] BinaryAutoencoder test passed!")


if __name__ == "__main__":
    test_binary_autoencoder()
