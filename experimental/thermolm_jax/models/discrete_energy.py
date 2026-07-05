"""
STATUS: EXPERIMENTAL / NOT VALIDATED. See STATUS.md. The energy network is
position-wise (no cross-token mixing) so it models no interactions, and the CD
loss here is sign-flipped (logsumexp(E_neg - E_pos)) which trains the model to
raise data energy. Kept for reference only.

Discrete Energy Function for ThermoLM JAX.

Implements energy function for discrete latent codes (FSQ outputs).
Based on D3PM (Discrete Denoising Diffusion Probabilistic Models).

Design Decision: D3PM for discrete diffusion
- Rationale: Proven approach for discrete diffusion, matches NVIDIA EDLM
- Impact: Better training stability for discrete latents
- Trade-off: More complex than continuous diffusion
- Downstream: Direct comparison with NVIDIA EDLM

Author: Apuroop Mutyala
Date: April 14, 2026
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional
from dataclasses import dataclass
from .d3pm import q_xt


@dataclass
class DiscreteEnergyConfig:
    """Configuration for discrete energy function."""
    vocab_size: int = 50257  # GPT-2 vocab size
    d_model: int = 512  # Model dimension
    d_latent: int = 64  # Latent dimension
    n_levels: int = 8  # Number of quantization levels per dimension
    num_energy_layers: int = 6  # Number of energy function layers
    num_energy_heads: int = 8  # Number of attention heads
    max_seq_len: int = 128  # Maximum sequence length
    dropout: float = 0.1  # Dropout rate
    num_diffusion_timesteps: int = 1000  # Number of diffusion timesteps for D3PM


class DiscreteEnergyFunction(nn.Module):
    """
    Discrete energy function for FSQ latent codes.
    
    Takes discrete codes and outputs energy (negative log probability).
    Uses transformer architecture to model interactions between latent dimensions.
    """
    
    config: DiscreteEnergyConfig
    
    def setup(self):
        """Initialize discrete energy function components."""
        # Embed discrete codes
        self.code_embed = nn.Embed(
            num_embeddings=self.config.n_levels,
            features=self.config.d_model,
        )
        
        # Positional encoding
        self.pos_embed = nn.Embed(
            num_embeddings=self.config.max_seq_len * self.config.d_latent,
            features=self.config.d_model,
        )
        
        # MLP layers for energy function (simpler than transformer)
        self.mlp_layers = [
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
        
        # Output projection to energy (scalar per position)
        self.energy_proj = nn.Dense(1)
    
    def __call__(
        self,
        codes: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute energy for discrete codes.
        
        Args:
            codes: (batch, seq_len, d_latent) discrete codes
            mask: (batch, seq_len) attention mask (1 for valid, 0 for padding)
        
        Returns:
            energy: (batch,) total energy (sum over tokens)
        """
        batch_size, seq_len, d_latent = codes.shape
        
        # Process sequence and latent dimensions separately to preserve structure
        # First, embed codes: (batch, seq_len, d_latent) -> (batch, seq_len, d_latent, d_model)
        codes_expanded = codes[..., None]  # (batch, seq_len, d_latent, 1)
        code_embeddings = self.code_embed(codes_expanded.squeeze(-1))  # (batch, seq_len, d_latent, d_model)
        
        # Add positional embeddings
        seq_positions = jnp.arange(seq_len)  # (seq_len,)
        seq_pos_emb = self.pos_embed(seq_positions)  # (seq_len, d_model)
        seq_pos_emb = seq_pos_emb[None, :, None, :]  # (1, seq_len, 1, d_model)
        x = code_embeddings + seq_pos_emb
        
        # Add latent dimension positional encoding
        lat_positions = jnp.arange(d_latent)  # (d_latent,)
        lat_pos_emb = self.pos_embed(lat_positions + seq_len)  # (d_latent, d_model)
        lat_pos_emb = lat_pos_emb[None, None, :, :]  # (1, 1, d_latent, d_model)
        x = x + lat_pos_emb
        
        # Reshape for MLP: (batch, seq_len, d_latent, d_model) -> (batch, seq_len * d_latent, d_model)
        x = x.reshape(batch_size, seq_len * d_latent, self.config.d_model)
        
        # Apply MLP layers
        for i in range(self.config.num_energy_layers):
            mlp_out = self.mlp_layers[i](x)
            x = x + mlp_out
            x = self.layer_norms[i](x)
        
        # Final layer norm
        x = self.final_ln(x)
        
        # Project to energy per position
        energies = self.energy_proj(x)  # (batch, seq_len * d_latent, 1)
        energies = energies.squeeze(-1)  # (batch, seq_len * d_latent)
        
        # Reshape back to (batch, seq_len, d_latent)
        energies = energies.reshape(batch_size, seq_len, d_latent)
        
        # Sum over latent dimensions and sequence length to get total energy
        if mask is not None:
            # Apply mask: only sum over valid positions
            mask_expanded = mask[:, :, None]  # (batch, seq_len, 1)
            energies = energies * mask_expanded
            total_energy = jnp.sum(energies, axis=(1, 2))  # (batch,)
        else:
            total_energy = jnp.sum(energies, axis=(1, 2))  # (batch,)
        
        return total_energy
    
    def compute_energy_gradient(
        self,
        codes: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute energy and gradient with respect to codes.
        
        Args:
            codes: (batch, seq_len, d_latent) discrete codes
            mask: (batch, seq_len) attention mask
        
        Returns:
            energy: (batch,) total energy
            gradient: (batch, seq_len, d_latent) energy gradient
        """
        def energy_fn(c):
            return self(c, mask=mask)
        
        energy, grad = jax.value_and_grad(energy_fn)(codes)
        return energy, grad


class DiscreteEnergyLoss(nn.Module):
    """
    Loss function for training discrete energy model.
    
    Implements contrastive divergence and maximum likelihood training.
    """
    
    config: DiscreteEnergyConfig
    
    def setup(self):
        """Initialize loss components."""
        self.energy_fn = DiscreteEnergyFunction(self.config)
    
    def __call__(
        self,
        codes: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        n_negative_samples: int = 10,
        key: jax.random.PRNGKey = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute contrastive divergence loss with D3PM noise schedule.
        
        Args:
            codes: (batch, seq_len, d_latent) positive samples
            mask: (batch, seq_len) attention mask
            n_negative_samples: Number of negative samples
            key: PRNG key for sampling
        
        Returns:
            loss: Scalar loss
            energy: Energy of positive samples
        """
        # Compute energy of positive samples
        energy_pos = self.energy_fn(codes, mask=mask)

        # Generate negative samples using THRML sampler
        from .thrml_discrete import THRMLSampler, THRMLConfig

        # Convert to unary weights (negative energy)
        unary_weights = -self.energy_fn(codes)

        # Create THRML sampler
        config = THRMLConfig(
            n_levels=self.config.n_levels,
            n_samples=n_negative_samples,
            n_warmup=10,
            n_steps=10,
            steps_per_sample=1,
        )
        sampler = THRMLSampler(config)

        # Sample from model
        neg_codes, _ = sampler.sample(
            unary_weights=unary_weights,
            key=key if key is not None else jax.random.PRNGKey(0),
        )
        
        # Compute energy of negative samples
        # Reshape for vectorized computation
        batch_size, seq_len, d_latent = codes.shape
        neg_codes_flat = neg_codes.reshape(n_negative_samples * batch_size, seq_len, d_latent)
        if mask is not None:
            mask_expanded = jnp.tile(mask, (n_negative_samples, 1))  # (n_neg * batch, seq_len)
        else:
            mask_expanded = None
        energy_neg_flat = self.energy_fn(neg_codes_flat, mask=mask_expanded)
        energy_neg = energy_neg_flat.reshape(n_negative_samples, batch_size)
        
        # Contrastive divergence loss: log(1 + exp(energy_neg - energy_pos))
        energy_pos_expanded = energy_pos[None, :]  # (1, batch)
        energy_diff = energy_neg - energy_pos_expanded  # (n_neg, batch)
        loss = jnp.mean(jax.nn.logsumexp(energy_diff, axis=0))
        
        return loss, jnp.mean(energy_pos)


def sample_discrete_energy(
    energy_fn: callable,
    shape: Tuple[int, int, int],
    n_levels: int,
    key: jax.random.PRNGKey,
    n_steps: int = 100,
    temperature: float = 1.0,
    mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample from discrete energy function using Gibbs sampling.
    
    Args:
        energy_fn: Energy function that takes (codes, mask) and returns energy
        shape: (batch_size, seq_len, d_latent)
        n_levels: Number of quantization levels
        key: PRNG key
        n_steps: Number of Gibbs sampling steps
        temperature: Sampling temperature
        mask: Optional attention mask
    
    Returns:
        samples: Sampled codes (batch_size, seq_len, d_latent)
        energies: Energy at each step (n_steps,)
    """
    batch_size, seq_len, d_latent = shape
    
    # Initialize with random samples
    key, init_key = jax.random.split(key)
    samples = jax.random.randint(
        init_key,
        shape=shape,
        minval=0,
        maxval=n_levels,
    )
    
    energies = []
    
    # Gibbs sampling
    for step in range(n_steps):
        key, step_key = jax.random.split(key)
        
        # Sample each position sequentially
        # (For efficiency, could use block sampling like in thrml_discrete.py)
        for seq_idx in range(seq_len):
            for lat_idx in range(d_latent):
                key, pos_key = jax.random.split(key)
                
                # Compute energy for all possible values at this position
                def energy_for_level(level):
                    test_samples = samples.at[:, seq_idx, lat_idx].set(level)
                    return energy_fn(test_samples, mask)
                
                all_levels = jnp.arange(n_levels)
                energies_for_levels = jax.vmap(energy_for_level)(all_levels)
                energies_for_levels = energies_for_levels.T  # (batch, n_levels)
                
                # Apply temperature
                energies_scaled = energies_for_levels / temperature
                
                # Compute probabilities
                log_probs = -energies_scaled
                log_probs = log_probs - jax.nn.logsumexp(log_probs, axis=1, keepdims=True)
                probs = jnp.exp(log_probs)
                
                # Sample new value
                new_value = jax.random.categorical(pos_key, probs, axis=1)
                samples = samples.at[:, seq_idx, lat_idx].set(new_value.squeeze())
        
        # Compute energy
        energy = energy_fn(samples, mask)
        energies.append(energy)
    
    return samples, jnp.array(energies)
