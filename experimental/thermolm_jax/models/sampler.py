"""
STATUS: EXPERIMENTAL / NOT VALIDATED. See STATUS.md. Despite the name and
docstrings, this does NOT use THRML (the THRML imports are unused) and is not a
correct Gibbs sampler: it is Metropolis–Hastings with a fixed N(0,I)
independence proposal, missing the proposal ratio and reusing the RNG key, so it
does not target exp(-E). Use thermolm_jax.sampling.chromatic_gibbs instead.

THRML Sampler Module for ThermoLM JAX

Implements thermodynamic sampling using THRML library.
Provides block Gibbs sampling with graph coloring for parallelism.

Author: Apuroop Mutyala
Date: April 2026
"""

import jax
import jax.numpy as jnp
from typing import Optional, Dict, Any, List, Callable
import thrml
from thrml import Block, BlockGibbsSpec, SamplingSchedule, sample_states, CategoricalNode


class THRMLSamplerJAX:
    """
    THRML-compatible block Gibbs sampler.
    
    This sampler performs thermodynamic sampling from energy functions
    using block Gibbs sampling with graph coloring for parallelism.
    Designed for compatibility with Extropic's TSU hardware.
    
    Attributes:
        d_latent: Latent dimension
        num_steps: Number of MCMC steps
        temperature: Sampling temperature
    """
    
    def __init__(
        self,
        d_latent: int = 64,
        num_steps: int = 50,
        temperature: float = 1.0
    ):
        """
        Initialize THRML sampler.
        
        Args:
            d_latent: Latent dimension
            num_steps: Number of MCMC steps
            temperature: Sampling temperature
        """
        self.d_latent = d_latent
        self.num_steps = num_steps
        self.temperature = temperature
    
    def sample(
        self,
        z_init: jnp.ndarray,
        energy_fn: Callable,
        key: jax.random.PRNGKey,
        adjacency: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Sample using block Gibbs sampling.
        
        Args:
            z_init: Initial state (seq_len, d_latent)
            energy_fn: Energy function
            key: PRNG key
            adjacency: Sparse adjacency matrix for graph coloring
        
        Returns:
            z_samples: (num_steps, seq_len, d_latent) samples
        """
        # Compute blocks using graph coloring
        blocks = self._create_blocks(z_init.shape[0], adjacency)
        
        # Use jax.lax.scan for JIT compatibility (Phase 2.6)
        def scan_body(carry, subkey):
            z = carry
            z_new = self._gibbs_step(z, blocks, energy_fn, subkey)
            return z_new, z_new  # (carry, output)
        
        keys = jax.random.split(key, self.num_steps)
        z_final, all_samples = jax.lax.scan(scan_body, z_init, keys)
        
        # Include initial state in samples
        samples = jnp.concatenate([z_init[None, :, :], all_samples], axis=0)
        
        return samples
    
    def _create_blocks(
        self,
        seq_len: int,
        adjacency: Optional[jnp.ndarray] = None
    ) -> List[List[int]]:
        """
        Create blocks using graph coloring.
        
        Args:
            seq_len: Sequence length
            adjacency: Sparse adjacency matrix
        
        Returns:
            blocks: List of variable indices for each color class
        """
        if adjacency is None:
            # Default: alternating blocks
            blocks = [
                list(range(0, seq_len, 2)),  # Even indices
                list(range(1, seq_len, 2))   # Odd indices
            ]
        else:
            # Use graph coloring algorithm
            blocks = self._graph_coloring(adjacency)
        
        return blocks
    
    def _graph_coloring(self, adjacency: jnp.ndarray) -> List[List[int]]:
        """
        Perform graph coloring for parallel sampling.
        
        Args:
            adjacency: Adjacency matrix
        
        Returns:
            blocks: List of variable indices for each color class
        """
        # Greedy graph coloring (Welsh-Powell algorithm)
        seq_len = adjacency.shape[0]
        
        # Sort vertices by degree (descending)
        degrees = jnp.sum(adjacency, axis=1)
        vertices = jnp.argsort(degrees)[::-1]
        
        # Convert to Python list for graph coloring (not JIT-traced)
        vertices_list = vertices.tolist()
        adjacency_list = adjacency.tolist()
        
        colors = {}
        for vertex in vertices_list:
            # Find used colors among neighbors
            neighbor_colors = set()
            for neighbor in range(seq_len):
                if adjacency_list[vertex][neighbor] > 0 and neighbor in colors:
                    neighbor_colors.add(colors[neighbor])
            
            # Assign smallest available color
            color = 0
            while color in neighbor_colors:
                color += 1
            colors[vertex] = color
        
        # Group by color
        blocks = {}
        for vertex, color in colors.items():
            blocks.setdefault(color, []).append(vertex)
        
        return list(blocks.values())
    
    def _gibbs_step(
        self,
        z: jnp.ndarray,
        blocks: List[List[int]],
        energy_fn: Callable,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """
        Perform one Gibbs sampling step.
        
        Args:
            z: Current state
            blocks: Variable blocks
            energy_fn: Energy function
            key: PRNG key
        
        Returns:
            z_new: Updated state
        """
        z_new = z.copy()
        
        # Update each block sequentially (blocks can be updated in parallel)
        for block_idx, block in enumerate(blocks):
            key, subkey = jax.random.split(key)
            # Update all variables in block in parallel using vmap
            block_indices = jnp.array(block)
            
            def update_single(i, z_current, subkey_i):
                return self._sample_conditional(z_current, i, energy_fn, subkey_i)
            
            # Split keys for each variable in block
            subkeys = jax.random.split(subkey, len(block))
            
            # Update each variable in block
            for i, subkey_i in zip(block, subkeys):
                z_new = z_new.at[i].set(self._sample_conditional(
                    z_new, i, energy_fn, subkey_i
                ))
        
        return z_new
    
    def _sample_conditional(
        self,
        z: jnp.ndarray,
        i: int,
        energy_fn: Callable,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """
        Sample from conditional distribution p(z_i | z_{-i}).
        
        Args:
            z: Current state
            i: Variable index
            energy_fn: Energy function
            key: PRNG key
        
        Returns:
            z_i_new: New value for variable i
        """
        # Metropolis-Hastings acceptance
        z_proposed = z.at[i].set(jax.random.normal(key, shape=(self.d_latent,)))
        
        E_current = energy_fn(z)
        E_proposed = energy_fn(z_proposed)
        delta_E = E_proposed - E_current
        
        # Use jnp.where for JIT compatibility (no Python if on JAX arrays)
        accept_prob = jnp.where(delta_E < 0, 1.0, jnp.exp(-delta_E / self.temperature))
        accept = jax.random.uniform(key) < accept_prob
        
        return jnp.where(accept, z_proposed[i], z[i])


def metropolis_hastings(
    z: jnp.ndarray,
    i: int,
    energy_fn: Callable,
    temperature: float,
    key: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Metropolis-Hastings acceptance for Gibbs sampling.
    
    Args:
        z: Current state
        i: Variable index
        energy_fn: Energy function
        temperature: Temperature
        key: PRNG key
    
    Returns:
        z_new: New state
    """
    # Propose new value
    z_proposed = z.at[i].set(jax.random.normal(key, shape=z[i].shape))
    
    # Compute energy difference
    E_current = energy_fn(z)
    E_proposed = energy_fn(z_proposed)
    delta_E = E_proposed - E_current
    
    # Accept/reject using jnp.where for JIT compatibility
    accept_prob = jnp.where(delta_E < 0, 1.0, jnp.exp(-delta_E / temperature))
    accept = jax.random.uniform(key) < accept_prob
    
    return jnp.where(accept, z_proposed, z)


# Note: For full THRML BlockSamplingProgram integration, see thrml_discrete.py
# This sampler provides a general-purpose JAX-compatible Gibbs sampler
# THRML-specific sampling with TSU hardware acceleration is available in THRMLSampler
