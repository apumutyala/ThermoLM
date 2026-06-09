"""
Hybrid EDLM Model for Phase 2B.

Integrates continuous encoder, quantization layer, and hybrid energy function
into a complete hybrid continuous-discrete model for two-stage training.

Design Decision: Hybrid model for two-stage training
- Rationale: Train continuous model first, then quantize for TSU compatibility
- Impact: Better training stability and TSU deployment
- Trade-off: More complex training pipeline
- Downstream: Enables comparison of continuous vs discrete performance

Author: Apuroop Mutyala
Date: April 15, 2026
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from thermolm_jax.models.continuous_encoder import (
    ContinuousEncoder,
    ContinuousDecoder,
    ContinuousEncoderConfig,
)
from thermolm_jax.models.quantization import (
    QuantizationLayer,
    QuantizationConfig,
)
from thermolm_jax.models.hybrid_energy import (
    HybridEnergyFunction,
    HybridEnergyLoss,
    HybridEnergyConfig,
)


@dataclass
class HybridEDLMConfig:
    """Configuration for hybrid EDLM model."""
    # Encoder config
    vocab_size: int = 50257
    d_model: int = 512
    d_latent: int = 64
    num_encoder_layers: int = 6
    num_encoder_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 128
    dropout: float = 0.1
    
    # Quantization config
    n_levels: int = 8
    quantization_type: str = "fsq"  # "fsq", "vq", "learned", "straight_through"
    commitment_cost: float = 0.25
    
    # Energy function config
    num_energy_layers: int = 6
    num_energy_heads: int = 8
    continuous_weight: float = 0.5
    discrete_weight: float = 0.5


class HybridEDLM(nn.Module):
    """
    Hybrid EDLM model combining continuous and discrete components.
    
    Supports two-stage training:
    - Stage 1: Train continuous encoder + continuous energy
    - Stage 2: Freeze encoder, train quantization + discrete energy
    """
    
    config: HybridEDLMConfig
    
    def setup(self):
        """Initialize hybrid EDLM components."""
        # Continuous encoder
        encoder_config = ContinuousEncoderConfig(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            d_latent=self.config.d_latent,
            num_layers=self.config.num_encoder_layers,
            num_heads=self.config.num_encoder_heads,
            d_ff=self.config.d_ff,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout,
        )
        self.encoder = ContinuousEncoder(encoder_config)
        
        # Continuous decoder (for reconstruction)
        self.decoder = ContinuousDecoder(encoder_config)
        
        # Quantization layer
        quantization_config = QuantizationConfig(
            d_latent=self.config.d_latent,
            n_levels=self.config.n_levels,
            quantization_type=self.config.quantization_type,
            commitment_cost=self.config.commitment_cost,
        )
        self.quantization = QuantizationLayer(quantization_config)
        
        # Hybrid energy function
        energy_config = HybridEnergyConfig(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            d_latent=self.config.d_latent,
            n_levels=self.config.n_levels,
            num_energy_layers=self.config.num_energy_layers,
            num_energy_heads=self.config.num_energy_heads,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout,
            continuous_weight=self.config.continuous_weight,
            discrete_weight=self.config.discrete_weight,
        )
        self.energy_fn = HybridEnergyFunction(energy_config)
        
        # Hybrid energy loss
        self.loss_fn = HybridEnergyLoss(energy_config)
    
    def encode(
        self,
        tokens: jnp.ndarray,
        quantize: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Optional[Dict[str, Any]]]:
        """
        Encode tokens to continuous latents (and optionally quantized codes).
        
        Args:
            tokens: (batch, seq_len) token IDs
            quantize: Whether to quantize latents
        
        Returns:
            latents: (batch, seq_len, d_latent) continuous latents
            codes: (batch, seq_len, d_latent) discrete codes (None if not quantized)
            quantized: (batch, seq_len, d_latent) quantized latents (None if not quantized)
            quantization_info: Dictionary with quantization info (None if not quantized)
        """
        # Encode to continuous latents
        latents, embeddings = self.encoder(tokens)
        
        if quantize:
            quantized, codes, quantization_info = self.quantization(latents, train=False)
            return latents, codes, quantized, quantization_info
        else:
            return latents, None, None, None
    
    def decode(
        self,
        latents: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Decode continuous latents to token logits.
        
        Args:
            latents: (batch, seq_len, d_latent) continuous latents
        
        Returns:
            logits: (batch, seq_len, vocab_size) token logits
        """
        return self.decoder(latents)
    
    def compute_energy(
        self,
        latents: Optional[jnp.ndarray] = None,
        codes: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Compute hybrid energy.
        
        Args:
            latents: (batch, seq_len, d_latent) continuous latents (optional)
            codes: (batch, seq_len, d_latent) discrete codes (optional)
            mask: (batch, seq_len) attention mask
        
        Returns:
            energy: (batch,) total energy
            info: Dictionary with component energies
        """
        return self.energy_fn(latents=latents, codes=codes, mask=mask)
    
    def compute_loss(
        self,
        latents: jnp.ndarray,
        codes: jnp.ndarray,
        quantization_info: Dict[str, Any],
        mask: Optional[jnp.ndarray] = None,
        key: Optional[jax.random.PRNGKey] = None,
        stage: int = 2,
        tokens: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Compute hybrid energy loss with optional reconstruction loss for Stage 1.
        
        Args:
            latents: (batch, seq_len, d_latent) continuous latents
            codes: (batch, seq_len, d_latent) discrete codes
            quantization_info: Dictionary with quantization loss
            mask: (batch, seq_len) attention mask
            key: PRNG key for sampling
            stage: Training stage (1 = continuous only, 2 = quantized)
            tokens: (batch, seq_len) original tokens for reconstruction loss (Stage 1 only)
        
        Returns:
            loss: Total loss
            info: Dictionary with loss components
        """
        if stage == 1 and tokens is not None:
            # Stage 1: Train continuous encoder with reconstruction loss
            # Decode latents back to token logits
            logits = self.decode(latents)
            
            # Compute reconstruction loss (cross-entropy)
            if mask is not None:
                # Apply mask
                mask_expanded = mask[:, :, None]
                logits = logits * mask_expanded
            
            # Cross-entropy loss
            recon_loss = optax.softmax_cross_entropy_with_integer_labels(logits, tokens)
            if mask is not None:
                recon_loss = (recon_loss * mask).sum() / mask.sum()
            else:
                recon_loss = recon_loss.mean()
            
            # Continuous energy loss for representation learning
            energy, energy_info = self.energy_fn(latents=latents, mask=mask)
            continuous_loss = jnp.mean(energy)
            
            # Total loss for Stage 1
            loss = recon_loss + 0.1 * continuous_loss
            
            info = {
                'total_loss': loss,
                'reconstruction_loss': recon_loss,
                'continuous_energy_loss': continuous_loss,
                'stage': 1,
            }
        else:
            # Stage 2: Train quantization and discrete energy
            loss, info = self.loss_fn(latents, codes, quantization_info, mask, key)
            info['stage'] = 2
        
        return loss, info
    
    def __call__(
        self,
        tokens: jnp.ndarray,
        quantize: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Optional[Dict[str, Any]]]:
        """
        Forward pass through hybrid model.
        
        Args:
            tokens: (batch, seq_len) token IDs
            quantize: Whether to quantize latents
        
        Returns:
            latents: (batch, seq_len, d_latent) continuous latents
            codes: (batch, seq_len, d_latent) discrete codes (None if not quantized)
            quantized: (batch, seq_len, d_latent) quantized latents (None if not quantized)
            quantization_info: Dictionary with quantization info (None if not quantized)
        """
        return self.encode(tokens, quantize=quantize)


# TODO: Implement progressive quantization (gradually increase n_levels during training) - Implemented below
# TODO: Add temperature annealing for energy function (improves convergence)


class ProgressiveQuantizationScheduler:
    """
    Scheduler for progressive quantization during training.
    
    Gradually increases the number of quantization levels from a low value
    to the target value, allowing the model to learn coarse-to-fine representations.
    """
    
    def __init__(
        self,
        initial_n_levels: int,
        target_n_levels: int,
        total_steps: int,
    ):
        """
        Initialize progressive quantization scheduler.
        
        Args:
            initial_n_levels: Starting number of quantization levels
            target_n_levels: Target number of quantization levels
            total_steps: Total training steps
        """
        self.initial_n_levels = initial_n_levels
        self.target_n_levels = target_n_levels
        self.total_steps = total_steps
    
    def get_n_levels(self, step: int) -> int:
        """
        Get current number of quantization levels.
        
        Args:
            step: Current training step
        
        Returns:
            n_levels: Current number of quantization levels
        """
        if step >= self.total_steps:
            return self.target_n_levels
        
        # Linear interpolation
        progress = step / self.total_steps
        n_levels = int(
            self.initial_n_levels + 
            progress * (self.target_n_levels - self.initial_n_levels)
        )
        
        return max(self.initial_n_levels, min(self.target_n_levels, n_levels))


class TemperatureAnnealingScheduler:
    """
    Scheduler for temperature annealing during training.
    
    Gradually decreases temperature during training to improve convergence
    of energy-based models.
    """
    
    def __init__(
        self,
        initial_temperature: float,
        final_temperature: float,
        total_steps: int,
        annealing_type: str = "linear",
    ):
        """
        Initialize temperature annealing scheduler.
        
        Args:
            initial_temperature: Starting temperature
            final_temperature: Final temperature
            total_steps: Total training steps
            annealing_type: Type of annealing ("linear", "exponential", "cosine")
        """
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.total_steps = total_steps
        self.annealing_type = annealing_type
    
    def get_temperature(self, step: int) -> float:
        """
        Get current temperature.
        
        Args:
            step: Current training step
        
        Returns:
            temperature: Current temperature
        """
        if step >= self.total_steps:
            return self.final_temperature
        
        progress = step / self.total_steps
        
        if self.annealing_type == "linear":
            temperature = self.initial_temperature + progress * (self.final_temperature - self.initial_temperature)
        elif self.annealing_type == "exponential":
            temperature = self.initial_temperature * (self.final_temperature / self.initial_temperature) ** progress
        elif self.annealing_type == "cosine":
            temperature = self.initial_temperature + 0.5 * (self.final_temperature - self.initial_temperature) * (1 - jnp.cos(jnp.pi * progress))
        else:
            raise ValueError(f"Unknown annealing type: {self.annealing_type}")
        
        return float(temperature)
