"""
STATUS: EXPERIMENTAL / NOT VALIDATED. See STATUS.md. The trained energy is not
the distribution actually sampled here (the sampler call signatures are
mismatched and a deep-net energy is not representable as THRML factors), so
generation does not depend on the learned energy. Kept for reference only.

Discrete EDLM Model Integration.

Integrates FSQ encoder, discrete energy function, and THRML sampler
into a complete discrete Energy-Based Diffusion Language Model.

Design Decision: Modular discrete EDLM
- Rationale: Clear separation of encoding, energy, and sampling components
- Impact: Easy to swap individual components (e.g., different quantization)
- Trade-off: More complex than monolithic model
- Downstream: Direct comparison with NVIDIA EDLM

Author: Apuroop Mutyala
Date: April 14, 2026
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .fsq import FSQEncoder, FSQDecoder, FSQConfig
from .discrete_energy import DiscreteEnergyFunction, DiscreteEnergyConfig
from .thrml_discrete import THRMLSampler, THRMLConfig


@dataclass
class DiscreteEDLMConfig:
    """Configuration for discrete EDLM model."""
    vocab_size: int = 50257  # GPT-2 vocab size
    d_model: int = 512  # Model dimension
    d_latent: int = 64  # Latent dimension
    n_levels: int = 8  # Number of quantization levels per dimension
    max_seq_len: int = 128  # Maximum sequence length
    # Energy function config
    num_energy_layers: int = 6
    num_energy_heads: int = 8
    dropout: float = 0.1
    # Sampling config
    block_size: int = 16
    n_samples: int = 10
    n_steps: int = 100
    temperature: float = 1.0
    use_hardware: bool = False


class DiscreteEDLM(nn.Module):
    """
    Discrete Energy-Based Diffusion Language Model.
    
    Combines:
    - FSQ encoder: tokens -> discrete latent codes
    - Discrete energy function: codes -> energy
    - THRML sampler: energy -> sampled codes
    - FSQ decoder: codes -> tokens
    """
    
    config: DiscreteEDLMConfig
    
    def setup(self):
        """Initialize discrete EDLM components."""
        # Create sub-configs
        fsq_config = FSQConfig(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            d_latent=self.config.d_latent,
            n_levels=self.config.n_levels,
            max_seq_len=self.config.max_seq_len,
        )

        energy_config = DiscreteEnergyConfig(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            d_latent=self.config.d_latent,
            n_levels=self.config.n_levels,
            num_energy_layers=self.config.num_energy_layers,
            num_energy_heads=self.config.num_energy_heads,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout,
        )

        sampler_config = THRMLConfig(
            vocab_size=self.config.vocab_size,
            d_latent=self.config.d_latent,
            n_levels=self.config.n_levels,
            block_size=self.config.block_size,
            n_samples=self.config.n_samples,
            n_warmup=self.config.n_steps,
            n_steps=self.config.n_steps,
            steps_per_sample=2,
            temperature=self.config.temperature,
            use_hardware=self.config.use_hardware,
        )

        # Initialize components
        self.fsq_encoder = FSQEncoder(fsq_config)
        self.fsq_decoder = FSQDecoder(fsq_config)
        self.energy_fn = DiscreteEnergyFunction(energy_config)
        self.sampler = THRMLSampler(sampler_config)

        # Language model head to convert embeddings to logits
        self.lm_head = nn.Dense(self.config.vocab_size)

        # Store configs for later use
        self._fsq_config = fsq_config
        self._energy_config = energy_config
        self._sampler_config = sampler_config
    
    def encode(
        self,
        tokens: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode tokens to discrete latent codes.
        
        Args:
            tokens: (batch, seq_len) token IDs
        
        Returns:
            codes: (batch, seq_len, d_latent) discrete codes
            latents: (batch, seq_len, d_latent) continuous latents
        """
        return self.fsq_encoder(tokens)
    
    def decode(
        self,
        codes: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Decode discrete codes to embeddings.
        
        Args:
            codes: (batch, seq_len, d_latent) discrete codes
        
        Returns:
            embeddings: (batch, seq_len, d_model) continuous embeddings
        """
        return self.fsq_decoder(codes)
    
    def compute_energy(
        self,
        codes: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute energy of discrete codes.
        
        Args:
            codes: (batch, seq_len, d_latent) discrete codes
            mask: (batch, seq_len) attention mask
        
        Returns:
            energy: (batch,) total energy
        """
        return self.energy_fn(codes, mask=mask)
    
    def sample_codes(
        self,
        initial_state: Optional[jnp.ndarray] = None,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Sample discrete codes using THRML sampler.
        
        Args:
            initial_state: Optional initial state
            key: PRNG key for randomness
        
        Returns:
            samples: (batch, seq_len, d_latent) sampled codes
            info: Sampling information
        """
        return self.sampler.sample(self.energy_fn, initial_state, key)
    
    def generate(
        self,
        initial_tokens: Optional[jnp.ndarray] = None,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Generate text by sampling from the model.

        Args:
            initial_tokens: Optional initial tokens for conditioning
            key: PRNG key for randomness

        Returns:
            tokens: (batch, seq_len) generated token IDs
            info: Generation information
        """
        # Encode initial tokens if provided
        if initial_tokens is not None:
            codes, _, _ = self.fsq_encoder(initial_tokens)
            # Use as initial state for sampling
            initial_state = codes
        else:
            initial_state = None

        # Sample codes
        samples, sampling_info = self.sample_codes(initial_state, key)

        # Decode samples to embeddings
        embeddings = self.fsq_decoder(samples)

        # Convert embeddings to logits using language model head
        logits = self.lm_head(embeddings)  # (batch, seq_len, vocab_size)

        # Sample token IDs from logits
        if key is not None:
            key, sample_key = jax.random.split(key)
            tokens = jax.random.categorical(sample_key, logits, axis=-1)  # (batch, seq_len)
        else:
            # Use argmax if no key provided
            tokens = jnp.argmax(logits, axis=-1)  # (batch, seq_len)

        info = {
            'sampling_info': sampling_info,
            'embeddings': embeddings,
            'logits': logits,
        }

        return tokens, info
    
    def __call__(
        self,
        tokens: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Full encode-decode pass (for training).
        
        Args:
            tokens: (batch, seq_len) token IDs
        
        Returns:
            codes: (batch, seq_len, d_latent) discrete codes
            latents: (batch, seq_len, d_latent) continuous latents
            recon_embeddings: (batch, seq_len, d_model) reconstructed embeddings
        """
        codes, latents, recon_embeddings = self.fsq_encoder(tokens)
        return codes, latents, recon_embeddings


# TODO: Add language model head to convert embeddings to logits
# TODO: Implement conditional generation
