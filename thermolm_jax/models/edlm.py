"""
EDLM (Energy-Based Diffusion Language Model) for ThermoLM JAX

Implements the main EDLM model using DiT architecture with d3pm parameterization.
Based on NVIDIA's EDLM approach.

Author: Apuroop Mutyala
Date: April 2026
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Optional, Dict, Any, Tuple

from .energy_function import EnergyFunctionJAX
from .diffusion_schedule import DiffusionSchedule, cosine_schedule
from .d3pm import q_xt, d3pm_parameterization, d3pm_loss, sample_prior
from .timestep import TimestepEmbedder


class EDLM(nn.Module):
    """
    Energy-Based Diffusion Language Model.
    
    Integrates DiT energy function, d3pm diffusion, and sampling.
    
    Attributes:
        vocab_size: Size of the vocabulary
        hidden_size: Hidden dimension
        n_blocks: Number of DiT blocks
        n_heads: Number of attention heads
        cond_dim: Dimension of timestep conditioning
        num_timesteps: Number of diffusion timesteps
        dropout: Dropout rate
        mlp_ratio: MLP expansion ratio
    """
    
    vocab_size: int
    hidden_size: int = 512
    n_blocks: int = 6
    n_heads: int = 8
    cond_dim: int = 256
    num_timesteps: int = 1000
    dropout: float = 0.1
    mlp_ratio: float = 4.0
    
    def setup(self):
        """Initialize EDLM components."""
        # Energy function (DiT)
        self.energy_function = EnergyFunctionJAX(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            cond_dim=self.cond_dim,
            dropout=self.dropout,
            mlp_ratio=self.mlp_ratio,
        )
        
        # Diffusion schedule
        self.diffusion_schedule = cosine_schedule(self.num_timesteps)
        
        # Mask index (vocab_size if no mask token)
        self.mask_index = self.vocab_size  # Will be set by tokenizer
    
    def set_mask_index(self, mask_index: int) -> None:
        """
        Set mask token index.
        
        Args:
            mask_index: Index of mask token
        """
        self.mask_index = mask_index
    
    def __call__(
        self,
        x: jnp.ndarray,
        sigma: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Forward pass through EDLM.
        
        Args:
            x: Input tokens, shape (batch, seq_len)
            sigma: Noise level (timestep), shape (batch,)
            mask: Attention mask (optional), shape (batch, seq_len)
        
        Returns:
            Log probabilities, shape (batch, seq_len, vocab_size)
        """
        logits = self.energy_function(x, sigma, mask)
        
        # Apply d3pm parameterization
        logits = d3pm_parameterization(logits, self.mask_index)
        
        return logits
    
    def compute_loss(
        self,
        x0: jnp.ndarray,
        key: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute d3pm training loss.
        
        Args:
            x0: Clean tokens, shape (batch, seq_len)
            key: Random key
            mask: Attention mask (optional), shape (batch, seq_len)
        
        Returns:
            Dictionary with loss and other metrics
        """
        batch_size, seq_len = x0.shape
        
        # Sample timestep
        t = jax.random.uniform(key, (batch_size,), minval=0.0, maxval=1.0)
        
        # Get noise level (sigma)
        sigma, _ = self.diffusion_schedule(t)
        
        # Compute move chance
        move_chance = 1.0 - jnp.exp(-sigma)
        move_chance = move_chance[:, None]  # (batch, 1)
        
        # Compute noisy tokens
        xt = q_xt(x0, move_chance, self.mask_index)
        
        # Get model output
        logits = self(xt, sigma, mask)
        
        # Compute d3pm loss
        loss = d3pm_loss(logits, xt, x0, t, self.mask_index, self.num_timesteps)
        
        # Mask loss for padding
        if mask is not None:
            loss = loss * mask
        
        return {
            'loss': loss.mean(),
            'nll': loss.sum(),
            'token_mask': mask if mask is not None else jnp.ones_like(loss),
        }
    
    def sample(
        self,
        key: jnp.ndarray,
        batch_size: int,
        seq_len: int,
        num_steps: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Generate samples using reverse diffusion.
        
        Args:
            key: Random key
            batch_size: Batch size
            seq_len: Sequence length
            num_steps: Number of diffusion steps (optional)
        
        Returns:
            Generated tokens, shape (batch, seq_len)
        """
        # Use THRML block Gibbs sampling
        return thrml_block_gibbs_sample(
            self.energy_function,
            jax.random.normal(key, (batch_size, seq_len)),
            block_size=seq_len,
            key=key,
            n_steps=num_steps,
        )


def conditional_edlm_sample(
    energy_fn: callable,
    condition: jnp.ndarray,
    n_samples: int,
    key: jax.random.PRNGKey,
    n_steps: int = 100,
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample from EDLM with conditioning.
    
    Args:
        energy_fn: Energy function that takes (x, condition)
        condition: Conditioning information
        n_samples: Number of samples
        key: PRNG key
        n_steps: Number of sampling steps
        temperature: Sampling temperature
    
    Returns:
        samples: Generated samples
        energies: Energy values
    """
    # Initialize with noise
    key, init_key = jax.random.split(key)
    samples = jax.random.normal(init_key, (n_samples, condition.shape[-1]))
    
    # Langevin dynamics with conditioning
    def energy_cond(x):
        return energy_fn(x, condition)
    
    samples, energies = langevin_dynamics_sample(
        energy_cond,
        samples,
        key=key,
        n_steps=n_steps,
        temperature=temperature,
    )
    
    return samples, energies


def classifier_free_guidance_sample(
    energy_fn: callable,
    energy_uncond: callable,
    guidance_scale: float,
    n_samples: int,
    key: jax.random.PRNGKey,
    n_steps: int = 100,
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample with classifier-free guidance.
    
    Args:
        energy_fn: Conditional energy function
        energy_uncond: Unconditional energy function
        guidance_scale: Guidance scale
        n_samples: Number of samples
        key: PRNG key
        n_steps: Number of sampling steps
        temperature: Sampling temperature
    
    Returns:
        samples: Guided samples
        energies: Energy values
    """
    # Sample from both conditional and unconditional
    key, cond_key, uncond_key = jax.random.split(key, 3)
    
    samples_cond, _ = langevin_dynamics_sample(
        energy_fn,
        jax.random.normal(cond_key, (n_samples, 512)),
        key=cond_key,
        n_steps=n_steps,
        temperature=temperature,
    )
    
    samples_uncond, _ = langevin_dynamics_sample(
        energy_uncond,
        jax.random.normal(uncond_key, (n_samples, 512)),
        key=uncond_key,
        n_steps=n_steps,
        temperature=temperature,
    )
    
    # Apply guidance
    samples_guided = samples_uncond + guidance_scale * (samples_cond - samples_uncond)
    
    return samples_guided, energy_fn(samples_guided)


def remove_noise_penalty(
    energy_fn: callable,
    samples: jnp.ndarray,
    noise_level: float = 0.01,
) -> jnp.ndarray:
    """
    Remove noise penalty from final step energy.
    
    Args:
        energy_fn: Energy function
        samples: Samples to denoise
        noise_level: Noise level to remove
    
    Returns:
        denoised_samples: Samples with noise penalty removed
    """
    # Simple denoising by projecting to lower energy
    def denoise_step(x):
        grad = jax.grad(energy_fn)(x)
        return x - noise_level * grad
    
    denoised = jax.vmap(denoise_step)(samples)
    
    return denoised


def apply_latent_constraints(
    samples: jnp.ndarray,
    constraints: Dict[str, Any],
) -> jnp.ndarray:
    """
    Apply constraints to latent space samples.
    
    Args:
        samples: Latent samples
        constraints: Dictionary of constraints (e.g., min, max, norm)
    
    Returns:
        constrained_samples: Samples with constraints applied
    """
    constrained = samples
    
    if 'min' in constraints:
        constrained = jnp.maximum(constrained, constraints['min'])
    
    if 'max' in constraints:
        constrained = jnp.minimum(constrained, constraints['max'])
    
    if 'max_norm' in constraints:
        norm = jnp.linalg.norm(constrained, axis=-1, keepdims=True)
        scale = jnp.minimum(1.0, constraints['max_norm'] / (norm + 1e-8))
        constrained = constrained * scale
    
    return constrained


def thrml_block_gibbs_sample(
    energy_fn: callable,
    initial_state: jnp.ndarray,
    block_size: int,
    key: jax.random.PRNGKey,
    n_steps: int = 100,
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Sample using THRML-style block Gibbs sampling.
    
    This is an interface for THRML hardware integration.
    Delegates to THRMLSampler for actual THRML API usage.
    
    Args:
        energy_fn: Energy function
        initial_state: Initial state
        block_size: Size of blocks for Gibbs sampling
        key: PRNG key
        n_steps: Number of sampling steps
        temperature: Sampling temperature
    
    Returns:
        samples: Sampled states
        info: Sampling information
    """
    from .thrml_discrete import THRMLSampler, THRMLConfig
    
    # Create THRML sampler config
    config = THRMLConfig(
        n_levels=8,  # Default, should be parameterized
        n_samples=1,
        n_warmup=n_steps,
        n_steps=n_steps,
        steps_per_sample=1,
        temperature=temperature,
        block_size=block_size,
    )
    
    sampler = THRMLSampler(config)
    
    # Convert energy function to unary weights (simplified)
    # In a full implementation, this would properly extract factor weights
    # For now, use simple energy-based weights
    unary_weights = -energy_fn(initial_state)  # Negative energy as weights
    
    # Sample using THRML
    samples, info = sampler.sample(
        unary_weights=unary_weights,
        key=key,
    )
    
    return samples, info


def langevin_dynamics_sample(
    energy_fn: callable,
    initial_state: jnp.ndarray,
    key: jax.random.PRNGKey,
    n_steps: int = 100,
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample using Langevin dynamics.
    
    Args:
        energy_fn: Energy function
        initial_state: Initial state
        key: PRNG key
        n_steps: Number of sampling steps
        temperature: Sampling temperature
    
    Returns:
        samples: Sampled states
        energies: Energy values
    """
    # Initialize samples and energies
    samples = initial_state
    energies = energy_fn(samples)
    
    # Langevin dynamics
    for _ in range(n_steps):
        # Compute gradient
        grad = jax.grad(energy_fn)(samples)
        
        # Compute noise
        noise = jax.random.normal(key, samples.shape)
        
        # Update samples
        samples = samples - 0.5 * temperature * grad + jnp.sqrt(temperature) * noise
        
        # Compute new energies
        energies = energy_fn(samples)
    
    return samples, energies
