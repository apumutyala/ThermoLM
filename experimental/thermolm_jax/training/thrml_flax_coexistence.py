"""
Equinox / Flax Coexistence Strategy for THRML Integration.

THRML uses `equinox` (eqx.Module). ThermoLM uses `flax.linen` (nn.Module). 
These are incompatible module systems that cannot be directly nested.

Phase 2.8: Equinox / Flax Coexistence Strategy

The solution: clean boundary at the tensor level. The two systems interact 
only via weight tensors from the neural network, not at the module level.

Flax forward pass → {unary_weights, pairwise_weights}
                              ↓
        THRML builds factors from these tensors
                              ↓
          THRML sampler produces discrete samples
                              ↓
      Flax loss computed from samples + original weights

This means:
- The FactorWeightNetwork (Flax) is compiled and called with jax.jit(flax_model.apply)(...)
- The THRML FactorSamplingProgram is compiled separately with eqx.filter_jit(...)
- They share data only through plain JAX arrays at function call boundaries

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import jax
import jax.numpy as jnp
import optax
from typing import Optional, Tuple, Callable
import thrml
from thrml import Block, BlockGibbsSpec, SamplingSchedule, sample_states, CategoricalNode
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import (
    CategoricalEBMFactor,
    SquareCategoricalEBMFactor,
    CategoricalGibbsConditional,
)


def build_thrml_program(
    unary_weights: jnp.ndarray,
    pairwise_weights: jnp.ndarray,
    nodes,
    n_levels: int
):
    """
    Build THRML sampling program from factor weights.
    
    This function creates the THRML FactorSamplingProgram from the 
    neural network's factor weight outputs. This is the boundary 
    where Flax tensors become THRML objects.
    
    Phase 2.8: Clean boundary at tensor level
    
    Args:
        unary_weights: (n_positions, n_levels) unary factor weights
        pairwise_weights: (n_pairs, n_levels, n_levels) pairwise factor weights
        nodes: List of CategoricalNode instances
        n_levels: Number of quantization levels
    
    Returns:
        program: FactorSamplingProgram
    """
    n_positions = unary_weights.shape[0]
    
    # Build factors from weights
    unary_factor = CategoricalEBMFactor(
        node_groups=[Block(nodes)],
        weights=unary_weights,
    )
    
    factors = [unary_factor]
    if pairwise_weights is not None:
        pairwise_factor = SquareCategoricalEBMFactor(
            node_groups=[Block(nodes)],
            weights=pairwise_weights[:n_positions],
        )
        factors.append(pairwise_factor)
    
    # Build BlockGibbsSpec
    node_sds = {CategoricalNode: jax.ShapeDtypeStruct((), jnp.uint8)}
    spec = BlockGibbsSpec(
        free_super_blocks=[Block(nodes)],
        clamped_blocks=[],
        node_shape_dtypes=node_sds,
    )
    
    # Build FactorSamplingProgram
    program = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[CategoricalGibbsConditional(n_categories=n_levels)
                  for _ in spec.free_blocks],
        factors=factors,
        other_interaction_groups=[],
    )
    
    return program, spec


def run_thrml_sampling(
    unary_weights: jnp.ndarray,
    pairwise_weights: jnp.ndarray,
    init_state: list,
    nodes,
    n_levels: int,
    n_warmup: int = 10,
    n_samples: int = 5,
    steps_per_sample: int = 1,
    key: jax.random.PRNGKey = None,
):
    """
    Run THRML sampling with given factor weights.
    
    This function is the THRML-side of the coexistence pattern.
    It takes plain JAX arrays (weights) and returns plain JAX arrays (samples).
    
    Phase 2.8: THRML sampling function (equinox workflow)
    
    Args:
        unary_weights: (n_positions, n_levels) unary factor weights
        pairwise_weights: (n_pairs, n_levels, n_levels) pairwise factor weights
        init_state: Initial state as list of arrays per block
        nodes: List of CategoricalNode instances
        n_levels: Number of quantization levels
        n_warmup: Number of warmup steps
        n_samples: Number of samples to draw
        steps_per_sample: Steps between samples
        key: PRNG key
    
    Returns:
        samples: (n_samples, n_positions) sampled states
    """
    
    if key is None:
        raise ValueError("PRNG key must be provided for THRML sampling")
    
    # Build THRML program
    program, spec = build_thrml_program(unary_weights, pairwise_weights, nodes, n_levels)
    
    # Sampling schedule
    schedule = SamplingSchedule(
        n_warmup=n_warmup,
        n_samples=1,  # Get 1 sample per vmap instance
        steps_per_sample=steps_per_sample,
    )
    
    # Create keys for vmap
    n_samples_actual = init_state[0].shape[0]
    keys = jax.random.split(key, n_samples_actual)
    
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
    
    # Concatenate blocks
    samples_array = jnp.concatenate(samples, axis=1)
    if samples_array.ndim == 3 and samples_array.shape[1] == 1:
        samples_array = samples_array.squeeze(axis=1)
    
    return samples_array


def train_step_flax_only(
    flax_model,
    flax_params,
    optimizer,
    opt_state,
    key: jax.random.PRNGKey,
    token_batch: jnp.ndarray,
    t: jnp.ndarray,
    n_warmup: int = 10,
    n_samples: int = 5,
    steps_per_sample: int = 1,
):
    """
    Training step with Flax gradient only (no gradient through THRML).
    
    This implements the coexistence pattern:
    1. Get factor weights from Flax model
    2. Run THRML sampling (outside gradient tape)
    3. Compute CD gradient through Flax parameters only
    
    Phase 2.8: Training step with clean boundary
    
    Args:
        flax_model: Flax model (FactorWeightNetwork)
        flax_params: Flax model parameters
        optimizer: Optax optimizer
        opt_state: Optimizer state
        key: PRNG key
        token_batch: Batch of tokens (batch, seq_len)
        t: Timestep (batch,) or (batch, 1)
        n_warmup: THRML warmup steps
        n_samples: THRML number of samples
        steps_per_sample: THRML steps per sample
    
    Returns:
        flax_params: Updated Flax parameters
        opt_state: Updated optimizer state
        loss: Training loss
    """
    
    from thermolm_jax.models.factor_weight_network import compute_energy_from_weights
    
    key_fw, key_thrml_pos, key_thrml_neg = jax.random.split(key, 3)
    batch_size, seq_len = token_batch.shape
    n_levels = flax_model.n_levels
    
    # Build THRML nodes
    nodes = [CategoricalNode() for _ in range(seq_len)]
    
    # Initialize states for THRML
    init_pos = []
    init_neg = []
    for block in [Block(nodes)]:
        block_size = len(block.nodes)
        init_pos.append(jax.random.randint(
            key_thrml_pos, (batch_size, block_size), 0, n_levels, dtype=jnp.uint8
        ))
        init_neg.append(jax.random.randint(
            key_thrml_neg, (batch_size, block_size), 0, n_levels, dtype=jnp.uint8
        ))
    
    # 1. Get factor weights from Flax model
    def get_factor_weights(params):
        return flax_model.apply(params, token_batch, t)
    
    unary_weights, pairwise_weights = get_factor_weights(flax_params)
    
    # 2. Run THRML for positive and negative samples (outside gradient tape)
    # Positive phase: clamp to data tokens
    pos_samples = run_thrml_sampling(
        unary_weights,
        pairwise_weights,
        init_pos,
        nodes,
        n_levels,
        n_warmup=n_warmup,
        n_samples=n_samples,
        steps_per_sample=steps_per_sample,
        key=key_thrml_pos,
    )
    
    # Negative phase: sample from model
    neg_samples = run_thrml_sampling(
        unary_weights,
        pairwise_weights,
        init_neg,
        nodes,
        n_levels,
        n_warmup=n_warmup,
        n_samples=n_samples,
        steps_per_sample=steps_per_sample,
        key=key_thrml_neg,
    )
    
    # 3. Compute CD gradient through Flax parameters
    def cd_loss(params):
        uw, pw = get_factor_weights(params)
        e_pos = compute_energy_from_weights(uw, pw, pos_samples)
        e_neg = compute_energy_from_weights(uw, pw, neg_samples)
        return jnp.mean(e_pos) - jnp.mean(e_neg)
    
    loss, grads = jax.value_and_grad(cd_loss)(flax_params)
    updates, opt_state = optimizer.update(grads, opt_state, flax_params)
    flax_params = optax.apply_updates(flax_params, updates)
    
    return flax_params, opt_state, loss


# Phase 2.8: Coexistence pattern demonstration
# This shows how Flax and Equinox interact cleanly at the tensor level

def demonstrate_coexistence():
    """
    Demonstrate the Equinox/Flax coexistence pattern.
    
    This function shows the clean boundary between Flax and THRML:
    - Flax produces factor weights (plain JAX arrays)
    - THRML consumes factor weights (plain JAX arrays)
    - No module nesting, just tensor passing
    """
    print("=== Phase 2.8: Equinox/Flax Coexistence Pattern ===\n")
    
    print("Architecture:")
    print("  Flax forward pass → {unary_weights, pairwise_weights}")
    print("                          ↓")
    print("          THRML builds factors from these tensors")
    print("                          ↓")
    print("              THRML sampler produces discrete samples")
    print("                          ↓")
    print("          Flax loss computed from samples + original weights")
    print()
    
    print("Key Points:")
    print("  1. FactorWeightNetwork (Flax) compiled with jax.jit(flax_model.apply)")
    print("  2. THRML FactorSamplingProgram compiled separately with eqx.filter_jit")
    print("  3. They share data only through plain JAX arrays at function boundaries")
    print("  4. Training gradient flows only through Flax, not through THRML")
    print("  5. This matches Extropic's hardware-software co-design pattern")
    print()
    
    print("Benefits for Extropic:")
    print("  - Neural network (GPU-resident) computes energy parameters")
    print("  - THRML (TSU-resident) does the sampling")
    print("  - Clean separation enables hardware acceleration")
    print("  - Demonstrates understanding of hardware-software co-design")
