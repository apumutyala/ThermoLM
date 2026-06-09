"""
THRML Training Specification for Language EBMs.

Implements proper contrastive divergence training for language models using THRML.
Mirrors the IsingTrainingSpec pattern from thrml/models/ising.py.

Design Decision: THRML-based EBM training
- Rationale: Proper two-phase Monte Carlo gradient estimation for EBMs
- Impact: Correct training objective that minimizes KL(p_data || p_theta)
- Trade-off: Requires THRML library for sampling
- Downstream: Enables proper EBM training for Extropic's TSU hardware

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass
import thrml
from thrml import Block, BlockGibbsSpec, SamplingSchedule, sample_with_observation, CategoricalNode
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import CategoricalGibbsConditional


@dataclass
class LanguageModelTrainingConfig:
    """Configuration for language model training."""
    n_levels: int = 8  # Number of quantization levels
    n_warmup: int = 100  # Warmup steps for sampling
    n_samples: int = 10  # Number of samples to draw
    steps_per_sample: int = 2  # Steps between samples
    temperature: float = 1.0  # Sampling temperature


class LanguageModelTrainingSpec(eqx.Module):
    """
    Training specification for language EBM, mirroring IsingTrainingSpec.
    
    Implements two-phase Monte Carlo gradient estimation:
    - program_positive: THRML program with token nodes clamped (data-driven)
    - program_negative: THRML program with all nodes free (model distribution)
    
    This follows the pattern from thrml/models/ising.py for proper EBM training.
    """
    
    program_positive: FactorSamplingProgram
    program_negative: FactorSamplingProgram
    schedule_positive: SamplingSchedule
    schedule_negative: SamplingSchedule
    config: LanguageModelTrainingConfig
    
    def __init__(
        self,
        nodes: List[CategoricalNode],  # all CategoricalNodes
        token_blocks: List[Block],  # Blocks containing token-level nodes (clamped in positive)
        latent_blocks: List[Block],  # Blocks containing latent code nodes (sampled in both)
        factors: List,  # list of CategoricalEBMFactor / SquareCategoricalEBMFactor
        config: LanguageModelTrainingConfig,
    ):
        """
        Initialize language model training specification.
        
        Args:
            nodes: All CategoricalNodes in the model
            token_blocks: Blocks containing token-level nodes (clamped in positive phase)
            latent_blocks: Blocks containing latent code nodes (sampled in both phases)
            factors: List of THRML factors
            config: Training configuration
        """
        node_sds = {type(nodes[0]): jax.ShapeDtypeStruct((), jnp.uint8)}
        
        # Positive: token nodes clamped, only latent code nodes are free
        spec_pos = BlockGibbsSpec(latent_blocks, token_blocks, node_sds)
        self.program_positive = FactorSamplingProgram(
            spec_pos,
            [CategoricalGibbsConditional(n_categories=config.n_levels) 
             for _ in spec_pos.free_blocks],
            factors,
            []
        )
        
        # Negative: all nodes free
        all_blocks = latent_blocks + token_blocks  # as free superblocks
        spec_neg = BlockGibbsSpec(all_blocks, [], node_sds)
        self.program_negative = FactorSamplingProgram(
            spec_neg,
            [CategoricalGibbsConditional(n_categories=config.n_levels) 
             for _ in spec_neg.free_blocks],
            factors,
            []
        )
        
        self.schedule_positive = SamplingSchedule(
            n_warmup=config.n_warmup,
            n_samples=config.n_samples,
            steps_per_sample=config.steps_per_sample,
        )
        self.schedule_negative = SamplingSchedule(
            n_warmup=config.n_warmup,
            n_samples=config.n_samples,
            steps_per_sample=config.steps_per_sample,
        )
        
        self.config = config


@eqx.filter_jit
def estimate_lm_kl_grad(
    key: jax.random.PRNGKey,
    training_spec: LanguageModelTrainingSpec,
    token_data: jnp.ndarray,  # actual token indices for positive phase
    init_state_positive: List[jnp.ndarray],  # initial THRML state for positive chain
    init_state_negative: List[jnp.ndarray],  # initial THRML state for negative chain
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    """
    Two-phase Monte Carlo gradient for language EBM.
    
    This follows the pattern from estimate_kl_grad in thrml/models/ising.py.
    The gradient of KL(p_data || p_theta) is:
    
    ΔW = -(E_{p_data}[∂E_θ/∂W] - E_{p_θ}[∂E_θ/∂W])
       = -(⟨∂E_θ/∂W⟩_+ - ⟨∂E_θ/∂W⟩_-)
    
    - Positive phase: THRML samples latent codes conditioned on real tokens
    - Negative phase: THRML samples freely from p_theta
    
    Args:
        key: PRNG key
        training_spec: Training specification with positive/negative programs
        token_data: Actual token indices for positive phase (clamped values)
        init_state_positive: Initial THRML state for positive chain
        init_state_negative: Initial THRML state for negative chain
    
    Returns:
        pos_samples: Samples from positive phase (data-conditioned)
        neg_samples: Samples from negative phase (model distribution)
    """
    key_pos, key_neg = jax.random.split(key)
    
    # Positive phase: sample latent codes given real tokens
    pos_samples, _ = sample_with_observation(
        key_pos,
        training_spec.program_positive,
        training_spec.schedule_positive,
        init_state_positive,
        [token_data],  # clamped values for token blocks
        None, None,  # no observer
    )
    
    # Negative phase: free sampling from model
    neg_samples, _ = sample_with_observation(
        key_neg,
        training_spec.program_negative,
        training_spec.schedule_negative,
        init_state_negative,
        [],  # no clamped blocks
        None, None,
    )
    
    return pos_samples, neg_samples


def compute_ebm_loss(
    pos_energy: jnp.ndarray,
    neg_energy: jnp.ndarray,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """
    Compute EBM loss from positive and negative samples.
    
    The loss is: E(positive) - E(negative)
    This should be negative (model assigns lower energy to data than to its own samples).
    
    Args:
        pos_energy: Energy of positive samples (data-conditioned)
        neg_energy: Energy of negative samples (model distribution)
        temperature: Temperature parameter
    
    Returns:
        loss: Scalar EBM loss
    """
    # CD loss: E(positive) - E(negative) should be negative
    # (model assigns lower energy to data than to its own samples)
    pos_energy_scaled = pos_energy / temperature
    neg_energy_scaled = neg_energy / temperature
    
    cd_loss = jnp.mean(pos_energy_scaled) - jnp.mean(neg_energy_scaled)
    
    return cd_loss


def build_language_model_training_spec(
    unary_weights: jnp.ndarray,
    pairwise_weights: Optional[jnp.ndarray],
    n_levels: int,
    config: LanguageModelTrainingConfig,
) -> LanguageModelTrainingSpec:
    """
    Build a LanguageModelTrainingSpec from neural network weights.
    
    This is a convenience function that creates the necessary THRML components
    (nodes, blocks, factors) and wraps them in a training specification.
    
    Args:
        unary_weights: (n_positions, n_levels) unary factor weights from neural net
        pairwise_weights: (n_pairs, n_levels, n_levels) pairwise factor weights (optional)
        n_levels: Number of quantization levels
        config: Training configuration
    
    Returns:
        training_spec: LanguageModelTrainingSpec ready for training
    """
    
    n_positions = unary_weights.shape[0]
    
    # Build nodes (CategoricalNode doesn't take arguments, categories are determined by factors)
    nodes = [CategoricalNode() for _ in range(n_positions)]
    
    # Factors from neural net weights
    from thrml.models.discrete_ebm import CategoricalEBMFactor, SquareCategoricalEBMFactor
    
    # Unary factor: weights must be [n_nodes, n_categories]
    unary_factor = CategoricalEBMFactor(
        node_groups=[Block(nodes)],
        weights=unary_weights,
    )
    
    factors = [unary_factor]
    if pairwise_weights is not None:
        # Pairwise factor: weights must be [n_nodes, n_categories, n_categories]
        # Use single block for simplicity
        pairwise_factor = SquareCategoricalEBMFactor(
            node_groups=[Block(nodes)],
            weights=pairwise_weights[:n_positions],  # Ensure matching dimensions
        )
        factors.append(pairwise_factor)
    
    # For language models, we can treat all positions as latent codes
    # Token blocks would be the actual token embeddings (clamped in positive phase)
    # For simplicity, we'll split positions into token and latent blocks
    # In a real implementation, this would depend on the model architecture
    
    # Split: first half as "tokens" (clamped), second half as "latent" (sampled)
    split_point = n_positions // 2
    token_nodes = nodes[:split_point]
    latent_nodes = nodes[split_point:]
    
    token_blocks = [Block(token_nodes)]
    latent_blocks = [Block(latent_nodes)]
    
    return LanguageModelTrainingSpec(
        nodes=nodes,
        token_blocks=token_blocks,
        latent_blocks=latent_blocks,
        factors=factors,
        config=config,
    )
