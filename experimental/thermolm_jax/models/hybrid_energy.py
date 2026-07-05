"""
Hybrid Energy Function for Hybrid Continuous-Discrete Model.

Implements energy function that works with both continuous and discrete
representations for hybrid models.

Design Decision: Hybrid energy function
- Rationale: Enable two-stage training with continuous and discrete energy
- Impact: Better training stability and performance
- Trade-off: More complex than single-modality energy
- Downstream: Enables comparison of continuous vs discrete energy

Author: Apuroop Mutyala
Date: April 15, 2026
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class HybridEnergyConfig:
    """Configuration for hybrid energy function."""
    vocab_size: int = 50257
    d_model: int = 512
    d_latent: int = 64
    n_levels: int = 8
    num_energy_layers: int = 6
    num_energy_heads: int = 8
    max_seq_len: int = 128
    dropout: float = 0.1
    continuous_weight: float = 0.5  # Weight for continuous energy
    discrete_weight: float = 0.5  # Weight for discrete energy


class HybridEnergyFunction(nn.Module):
    """
    Hybrid energy function combining continuous and discrete energy.
    
    Computes energy as weighted sum of continuous and discrete energy components.
    """
    
    config: HybridEnergyConfig
    
    def setup(self):
        """Initialize hybrid energy function components."""
        # Continuous energy function (DiT-based)
        self.continuous_energy = nn.Sequential([
            nn.Dense(self.config.d_model),
            nn.gelu,
            nn.Dense(self.config.d_model),
            nn.gelu,
            nn.Dense(1),
        ])
        
        # Discrete energy function (from discrete_energy.py)
        # Code embedding
        self.code_embed = nn.Embed(
            num_embeddings=self.config.n_levels,
            features=self.config.d_model,
        )
        
        # Positional embedding
        self.pos_embed = nn.Embed(
            num_embeddings=self.config.max_seq_len * self.config.d_latent,
            features=self.config.d_model,
        )
        
        # MLP layers for discrete energy
        self.discrete_energy_layers = [
            nn.Sequential([
                nn.Dense(self.config.d_model),
                nn.gelu,
            ])
            for _ in range(self.config.num_energy_layers)
        ]
        
        # Layer norms
        self.layer_norms = [
            nn.LayerNorm(epsilon=1e-5)
            for _ in range(self.config.num_energy_layers)
        ]
        
        # Final layer norm
        self.final_ln = nn.LayerNorm(epsilon=1e-5)
        
        # Energy projection
        self.energy_proj = nn.Dense(1)
    
    def compute_continuous_energy(
        self,
        latents: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute continuous energy component.
        
        Args:
            latents: (batch, seq_len, d_latent) continuous latents
            mask: (batch, seq_len) attention mask
        
        Returns:
            energy: (batch,) continuous energy
        """
        # Flatten latents
        batch_size, seq_len, d_latent = latents.shape
        latents_flat = latents.reshape(batch_size, -1)  # (batch, seq_len * d_latent)
        
        # Compute continuous energy
        energy_cont = self.continuous_energy(latents_flat)  # (batch, 1)
        energy_cont = energy_cont.squeeze(-1)  # (batch,)
        
        return energy_cont
    
    def compute_discrete_energy(
        self,
        codes: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute discrete energy component.
        
        Args:
            codes: (batch, seq_len, d_latent) discrete codes
            mask: (batch, seq_len) attention mask
        
        Returns:
            energy: (batch,) discrete energy
        """
        batch_size, seq_len, d_latent = codes.shape
        
        # Flatten codes
        codes_flat = codes.reshape(batch_size, seq_len * d_latent)  # (batch, seq_len * d_latent)
        
        # Embed codes
        x = self.code_embed(codes_flat)  # (batch, seq_len * d_latent, d_model)
        
        # Add positional encoding
        positions = jnp.arange(seq_len * d_latent)
        pos_emb = self.pos_embed(positions)
        x = x + pos_emb[None, :, :]
        
        # Apply MLP layers
        for i in range(self.config.num_energy_layers):
            mlp_out = self.discrete_energy_layers[i](x)
            x = x + mlp_out
            x = self.layer_norms[i](x)
        
        # Final layer norm
        x = self.final_ln(x)
        
        # Project to energy per position
        energies = self.energy_proj(x)  # (batch, seq_len * d_latent, 1)
        energies = energies.squeeze(-1)  # (batch, seq_len * d_latent)
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = jnp.repeat(mask, d_latent, axis=1)  # (batch, seq_len * d_latent)
            energies = energies * mask_expanded
        
        # Sum over positions
        energy_disc = jnp.sum(energies, axis=1)  # (batch,)
        
        return energy_disc
    
    def __call__(
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
            energy: (batch,) total hybrid energy
            info: Dictionary with component energies
        """
        energy = 0.0
        info = {}
        
        # Compute continuous energy if latents provided
        if latents is not None:
            energy_cont = self.compute_continuous_energy(latents, mask)
            energy += self.config.continuous_weight * energy_cont
            info['continuous_energy'] = energy_cont
        
        # Compute discrete energy if codes provided
        if codes is not None:
            energy_disc = self.compute_discrete_energy(codes, mask)
            energy += self.config.discrete_weight * energy_disc
            info['discrete_energy'] = energy_disc
        
        return energy, info


class HybridEnergyLoss(nn.Module):
    """
    Loss function for hybrid energy model.
    
    Combines reconstruction loss, quantization loss, and contrastive loss.
    """
    
    config: HybridEnergyConfig
    
    def setup(self):
        """Initialize hybrid energy loss."""
        self.energy_fn = HybridEnergyFunction(self.config)
    
    def __call__(
        self,
        latents: jnp.ndarray,
        codes: jnp.ndarray,
        quantization_info: Dict[str, Any],
        mask: Optional[jnp.ndarray] = None,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Compute hybrid energy loss.
        
        Args:
            latents: (batch, seq_len, d_latent) continuous latents
            codes: (batch, seq_len, d_latent) discrete codes
            quantization_info: Dictionary with quantization loss
            mask: (batch, seq_len) attention mask
            key: PRNG key for sampling
        
        Returns:
            loss: Total loss
            info: Dictionary with loss components
        """
        # Compute energy of positive samples
        energy_pos, energy_info = self.energy_fn(latents=latents, codes=codes, mask=mask)
        
        # Sample negative codes for contrastive loss
        if key is not None:
            key, sample_key = jax.random.split(key)
            n_negative_samples = 10
            neg_codes = jax.random.randint(
                sample_key,
                shape=(n_negative_samples,) + codes.shape,
                minval=0,
                maxval=self.config.n_levels,
            )
            
            # Compute energy of negative samples
            batch_size, seq_len, d_latent = codes.shape
            neg_codes_flat = neg_codes.reshape(n_negative_samples * batch_size, seq_len, d_latent)
            if mask is not None:
                mask_expanded = jnp.tile(mask, (n_negative_samples, 1))
            else:
                mask_expanded = None
            
            energy_neg_flat, _ = self.energy_fn(codes=neg_codes_flat, mask=mask_expanded)
            energy_neg = energy_neg_flat.reshape(n_negative_samples, batch_size)
            
            # Contrastive divergence loss
            energy_pos_expanded = energy_pos[None, :]
            energy_diff = energy_neg - energy_pos_expanded
            contrastive_loss = jnp.mean(jax.nn.logsumexp(energy_diff, axis=0))
        else:
            contrastive_loss = 0.0
        
        # Quantization loss
        quantization_loss = quantization_info.get('commitment_loss', 0.0)
        
        # Total loss
        loss = contrastive_loss + quantization_loss
        
        info = {
            'total_loss': loss,
            'contrastive_loss': contrastive_loss,
            'quantization_loss': quantization_loss,
            'energy_pos': energy_pos,
        }
        info.update(energy_info)
        
        return loss, info


# TODO: Add adaptive weighting for continuous vs discrete energy - Implemented below
# TODO: Implement annealing schedule for weights - Implemented below


class AdaptiveWeightScheduler:
    """
    Scheduler for adaptive weighting between continuous and discrete energy.
    
    Dynamically adjusts the balance between continuous and discrete components
    during training to optimize representation learning.
    """
    
    def __init__(
        self,
        initial_continuous_weight: float,
        initial_discrete_weight: float,
        final_continuous_weight: float,
        final_discrete_weight: float,
        total_steps: int,
        schedule_type: str = "linear",
    ):
        """
        Initialize adaptive weight scheduler.
        
        Args:
            initial_continuous_weight: Starting weight for continuous energy
            initial_discrete_weight: Starting weight for discrete energy
            final_continuous_weight: Final weight for continuous energy
            final_discrete_weight: Final weight for discrete energy
            total_steps: Total training steps
            schedule_type: Type of schedule ("linear", "exponential", "cosine")
        """
        self.initial_continuous_weight = initial_continuous_weight
        self.initial_discrete_weight = initial_discrete_weight
        self.final_continuous_weight = final_continuous_weight
        self.final_discrete_weight = final_discrete_weight
        self.total_steps = total_steps
        self.schedule_type = schedule_type
    
    def get_weights(self, step: int) -> Tuple[float, float]:
        """
        Get current weights for continuous and discrete energy.
        
        Args:
            step: Current training step
        
        Returns:
            continuous_weight: Weight for continuous energy
            discrete_weight: Weight for discrete energy
        """
        if step >= self.total_steps:
            return self.final_continuous_weight, self.final_discrete_weight
        
        progress = step / self.total_steps
        
        if self.schedule_type == "linear":
            continuous_weight = (
                self.initial_continuous_weight + 
                progress * (self.final_continuous_weight - self.initial_continuous_weight)
            )
            discrete_weight = (
                self.initial_discrete_weight + 
                progress * (self.final_discrete_weight - self.initial_discrete_weight)
            )
        elif self.schedule_type == "exponential":
            continuous_weight = (
                self.initial_continuous_weight * 
                (self.final_continuous_weight / self.initial_continuous_weight) ** progress
            )
            discrete_weight = (
                self.initial_discrete_weight * 
                (self.final_discrete_weight / self.initial_discrete_weight) ** progress
            )
        elif self.schedule_type == "cosine":
            continuous_weight = (
                self.initial_continuous_weight + 
                0.5 * (self.final_continuous_weight - self.initial_continuous_weight) * 
                (1 - jnp.cos(jnp.pi * progress))
            )
            discrete_weight = (
                self.initial_discrete_weight + 
                0.5 * (self.final_discrete_weight - self.initial_discrete_weight) * 
                (1 - jnp.cos(jnp.pi * progress))
            )
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return float(continuous_weight), float(discrete_weight)
