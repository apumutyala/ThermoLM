"""
THRML Sampler for Discrete Energy Models.

Integrates Extropic's THRML library for discrete sampling on TSU hardware.
Provides a JAX-compatible interface for block Gibbs sampling using actual THRML API.

Design Decision: THRML integration for discrete sampling
- Rationale: Leverages Extropic's optimized hardware for discrete diffusion
- Impact: Faster sampling than software-based MCMC, direct TSU compatibility
- Trade-off: Requires THRML library (GPU simulation available)
- Downstream: Direct comparison with Extropic's DTM baseline

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import thrml
from thrml import Block, BlockGibbsSpec, SamplingSchedule, sample_states, CategoricalNode
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import (
    CategoricalEBMFactor,
    SquareCategoricalEBMFactor,
    CategoricalGibbsConditional,
)


@dataclass
class THRMLConfig:
    """Configuration for THRML sampler."""
    vocab_size: int = 50257  # GPT-2 vocab size
    d_latent: int = 64  # Latent dimension
    n_levels: int = 8  # Number of quantization levels
    block_size: int = 16  # Block size for Gibbs sampling
    n_samples: int = 10  # Number of samples to generate
    n_warmup: int = 100  # Number of warmup steps
    n_steps: int = 100  # Total number of sampling steps
    steps_per_sample: int = 1  # Steps between samples
    temperature: float = 1.0  # Sampling temperature
    use_hardware: bool = False  # Whether to use TSU hardware


class THRMLSampler:
    """
    THRML sampler for discrete energy models.
    
    Performs block Gibbs sampling on discrete latent codes using actual THRML library.
    Uses GPU simulation when TSU hardware unavailable.
    """
    
    config: THRMLConfig
    
    def __init__(self, config: THRMLConfig):
        """Initialize THRML sampler."""
        self.config = config
    
    def sample(
        self,
        unary_weights: jnp.ndarray,
        pairwise_weights: Optional[jnp.ndarray] = None,
        initial_codes: Optional[jnp.ndarray] = None,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Sample from energy function using THRML.
        
        Args:
            unary_weights: (n_positions, n_levels) unary factor weights from neural net
            pairwise_weights: (n_pairs, n_levels, n_levels) pairwise factor weights (optional)
            initial_codes: Optional initial state (n_positions,) uint8
            key: PRNG key for randomness
        
        Returns:
            samples: (n_samples, n_positions) sampled states
            info: Dictionary of sampling information
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        
        return self._sample_with_thrml(unary_weights, pairwise_weights, initial_codes, key)
    
    def _sample_with_thrml(
        self,
        unary_weights: jnp.ndarray,
        pairwise_weights: Optional[jnp.ndarray],
        initial_codes: Optional[jnp.ndarray],
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Sample using actual THRML library with FactorSamplingProgram.
        
        This follows Phase 2-C corrections:
        - Uses CategoricalGibbsConditional (not abstract SoftmaxConditional)
        - Uses BlockGibbsSpec with node_shape_dtypes
        - Uses FactorSamplingProgram (not BlockSamplingProgram directly)
        - Uses CategoricalNode with n_categories
        
        Args:
            unary_weights: (n_positions, n_levels) unary factor weights
            pairwise_weights: (n_pairs, n_levels, n_levels) pairwise factor weights
            initial_codes: Optional initial state (n_positions,) uint8
            key: PRNG key
        
        Returns:
            samples: (n_samples, n_positions) sampled states
            info: Dictionary of sampling information
        """
        n_positions = unary_weights.shape[0]
        n_levels = self.config.n_levels
        
        # Build nodes (CategoricalNode doesn't take arguments, categories determined by factors)
        nodes = [CategoricalNode() for _ in range(n_positions)]
        
        # For unary factor: weights must have shape [n_nodes, n_categories]
        # unary_weights is [n_positions, n_levels], which matches [n_nodes, n_categories]
        unary_factor = CategoricalEBMFactor(
            node_groups=[Block(nodes)],
            weights=unary_weights,
        )
        
        factors = [unary_factor]
        if pairwise_weights is not None:
            # For pairwise factor: weights must have shape [n_nodes_in_first_block, n_categories, n_categories]
            # pairwise_weights should be [n_positions-1, n_levels, n_levels]
            # But node_groups must have same number of nodes in each block
            # Simplified: use single block for all nodes
            pairwise_factor = SquareCategoricalEBMFactor(
                node_groups=[Block(nodes)],
                weights=pairwise_weights[:n_positions],  # Ensure matching dimensions
            )
            factors.append(pairwise_factor)
        
        # BlockGibbsSpec with node_shape_dtypes (Phase 2-C.2)
        # Use single block for simplicity (can be parallelized later)
        node_sds = {CategoricalNode: jax.ShapeDtypeStruct((), jnp.uint8)}
        spec = BlockGibbsSpec(
            free_super_blocks=[Block(nodes)],
            clamped_blocks=[],
            node_shape_dtypes=node_sds,
        )
        
        # FactorSamplingProgram (Phase 2-C.3)
        program = FactorSamplingProgram(
            gibbs_spec=spec,
            samplers=[CategoricalGibbsConditional(n_categories=n_levels)
                      for _ in spec.free_blocks],  # Phase 2-C.1
            factors=factors,
            other_interaction_groups=[],
        )
        
        # Initialize state
        # THRML expects state as list of arrays, one per block
        # Each array should have shape (n_batches, n_nodes_in_block) for vmap
        # Following pattern from THRML example
        if initial_codes is None:
            key, init_key = jax.random.split(key)
            # Create batched state for each block
            init_state = []
            for block in spec.free_blocks:
                block_size = len(block.nodes)
                init_key, subkey = jax.random.split(init_key)
                block_state = jax.random.randint(
                    subkey,
                    shape=(self.config.n_samples, block_size),
                    minval=0,
                    maxval=n_levels,
                    dtype=jnp.uint8,
                )
                init_state.append(block_state)
        else:
            # Broadcast and split by blocks
            initial_codes = jnp.tile(initial_codes[None, :], (self.config.n_samples, 1))
            init_state = []
            offset = 0
            for block in spec.free_blocks:
                block_size = len(block.nodes)
                block_state = initial_codes[:, offset:offset + block_size]
                init_state.append(block_state)
                offset += block_size
        
        # Run sampling
        # Use vmap pattern from THRML example for multiple samples
        schedule = SamplingSchedule(
            n_warmup=self.config.n_warmup,
            n_samples=1,  # Get 1 sample per vmap instance
            steps_per_sample=self.config.steps_per_sample,
        )
        
        # Create keys for vmap
        keys = jax.random.split(key, self.config.n_samples)
        
        # Use vmap to run multiple parallel sampling instances
        def sample_single(init_state_single, key_single):
            return sample_states(
                key_single,
                program,
                schedule,
                init_state_single,
                [],  # no clamped blocks
                [Block(nodes)],  # nodes to sample
            )
        
        samples = jax.vmap(sample_single)(init_state, keys)
        
        # samples is a list of arrays, one per block in nodes_to_sample
        # Each array has shape (n_samples, n_nodes_in_block)
        # Concatenate all blocks to get full state
        samples_array = jnp.concatenate(samples, axis=1)  # (n_samples, n_positions)
        # If samples has extra dimensions, squeeze them
        if len(samples_array.shape) == 3 and samples_array.shape[1] == 1:
            samples_array = samples_array.squeeze(axis=1)  # (n_samples, n_positions)
        
        info = {
            'n_warmup': self.config.n_warmup,
            'n_samples': self.config.n_samples,
            'steps_per_sample': self.config.steps_per_sample,
            'n_blocks': len(spec.free_blocks),
            'temperature': self.config.temperature,
            'hardware_used': self.config.use_hardware,
            'sampling_method': 'THRML FactorSamplingProgram with CategoricalGibbsConditional',
        }
    
        return samples_array, info
    
    
    





