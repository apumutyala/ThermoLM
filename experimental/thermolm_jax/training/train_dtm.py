"""
Training Loop for DTM

Implements the training loop for DTM using improved contrastive divergence,
adaptive correlation penalty, and total correlation penalty.

Design Decision: Modular Training Loop
- Rationale: Enables flexible training configurations
- Impact: Easy to experiment with different training strategies
- Trade-off: More complex than simple loop
- Downstream: Supports research experiments

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import jax
import jax.numpy as jnp
import optax
from typing import Callable, Optional, Dict, Tuple
from dataclasses import dataclass
import time

import equinox as eqx

from ..models.dtm import DTM, DTMConfig
from .contrastive_divergence import contrastive_divergence_loss, CDConfig


@dataclass
class TrainingConfig:
    """Configuration for DTM training."""
    n_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    
    # CD configuration
    cd_k: int = 1
    n_gibbs_steps: int = 100
    temperature: float = 1.0
    use_improved_cd: bool = True
    gradient_weight: float = 0.1
    
    # ACP configuration
    use_acp: bool = True
    target_autocorr: float = 0.03
    acp_lag: int = 100
    
    # TC configuration
    use_tc: bool = True
    lambda_tc: float = 0.1
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 50


class DTMTrainer:
    """
    Trainer for DTM models.
    
    Implements the training loop with:
    - Improved contrastive divergence
    - Adaptive correlation penalty
    - Total correlation penalty
    - Optimizer and learning rate scheduling
    """
    
    def __init__(
        self,
        model: DTM,
        config: TrainingConfig,
        key: jax.random.PRNGKey
    ):
        """Initialize trainer."""
        self.model = model
        self.config = config
        self.key = key
        
        # Initialize optimizer
        optimizer = optax.adamw(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.optimizer = optimizer
        # We train the model's (single, shared) quadratic EBM via CD.
        self.opt_state = optimizer.init(eqx.filter(model.ebm, eqx.is_array))

        # Initialize CD config
        self.cd_config = CDConfig(
            k=config.cd_k,
            n_gibbs_steps=config.n_gibbs_steps,
            temperature=config.temperature,
            l2_weight=config.gradient_weight,
        )

    def train_step(
        self,
        batch: jnp.ndarray,
        step: int
    ) -> Tuple[DTM, Dict[str, float]]:
        """
        Perform one contrastive-divergence training step on the model's EBM.

        Trains the (single, shared) quadratic EBM of the DTM via CD against the
        data batch. The negative phase is treated as constant (stop_gradient)
        inside ``contrastive_divergence_loss``, giving the correct two-term CD
        gradient.

        NOTE: the previous ACP / total-correlation penalties and the duplicated
        ``jax.grad`` call have been removed — the latter differentiated through
        the full sampler with a stale key and was incorrect. ACP/TC live in the
        (unvalidated) experimental modules.

        Args:
            batch: Data batch, shape (batch_size, n_vars), spins in {-1, +1}.
            step: Training step number (unused; kept for API stability).

        Returns:
            model, metrics
        """
        self.key, key_cd = jax.random.split(self.key)

        def loss_fn(ebm):
            return contrastive_divergence_loss(ebm, batch, key_cd, self.cd_config)

        (loss, info), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(self.model.ebm)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.model.ebm)
        new_ebm = eqx.apply_updates(self.model.ebm, updates)
        self.model = eqx.tree_at(lambda m: m.ebm, self.model, new_ebm)

        metrics = {
            "loss": float(loss),
            "cd_loss": float(info["cd_loss"]),
            "l2_term": float(info["l2_term"]),
            "E_data": float(info["E_data"]),
            "E_neg": float(info["E_neg"]),
            "total_loss": float(loss),
        }
        return self.model, metrics
    
    def train_epoch(
        self,
        data_loader: Callable,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            data_loader: Data loader yielding batches
            epoch: Epoch number
        
        Returns:
            epoch_metrics: Average metrics for the epoch
        """
        epoch_metrics = []
        
        for step, batch in enumerate(data_loader):
            self.model, metrics = self.train_step(batch, epoch * 1000 + step)
            epoch_metrics.append(metrics)
            
            if step % self.config.log_interval == 0:
                print(f"Epoch {epoch}, Step {step}: {metrics}")
        
        # Compute average metrics
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
        
        return avg_metrics


def train_dtm(
    data_loader: Callable,
    config: DTMConfig,
    training_config: TrainingConfig,
    key: jax.random.PRNGKey
) -> Tuple[DTM, Dict]:
    """
    Train a DTM model.
    
    Args:
        data_loader: Data loader yielding batches
        config: DTM configuration
        training_config: Training configuration
        key: Random key
    
    Returns:
        model: Trained model
        history: Training history
    """
    # Initialize model
    model = DTM(config, key)
    
    # Initialize trainer
    trainer = DTMTrainer(model, training_config, key)
    
    # Training loop
    history = []
    for epoch in range(training_config.n_epochs):
        print(f"\n=== Epoch {epoch} ===")
        epoch_metrics = trainer.train_epoch(data_loader, epoch)
        history.append(epoch_metrics)
        
        print(f"Epoch {epoch} metrics: {epoch_metrics}")
    
    return trainer.model, history


def test_training_loop():
    """Test training loop implementation."""
    print("Testing DTM training loop...")
    
    # Simple data loader
    def simple_data_loader():
        key = jax.random.PRNGKey(0)
        for _ in range(10):
            batch = jax.random.randint(key, (4, 64), minval=0, maxval=2) * 2 - 1
            yield batch
    
    # Initialize model
    dtm_config = DTMConfig(
        n_data_vars=64,
        n_latent_vars=16,
        T=100
    )
    key = jax.random.PRNGKey(0)
    model = DTM(dtm_config, key)
    
    # Initialize trainer
    training_config = TrainingConfig(
        n_epochs=2,
        batch_size=4,
        n_gibbs_steps=10,
        use_acp=False,
        use_tc=False
    )
    trainer = DTMTrainer(model, training_config, key)
    
    # Test single training step
    batch = jax.random.randint(key, (4, 80), minval=0, maxval=2) * 2 - 1
    model, metrics = trainer.train_step(batch, step=0)
    
    print(f"Training step metrics: {metrics}")
    assert "loss" in metrics
    assert "total_loss" in metrics
    
    print("[SUCCESS] Training loop test passed!")


if __name__ == "__main__":
    test_training_loop()
