"""
Two-stage training script for hybrid EDLM model.

Implements two-stage training:
- Stage 1: Train continuous encoder + continuous energy
- Stage 2: Freeze encoder, train quantization + discrete energy

Design Decision: Two-stage training for hybrid model
- Rationale: Train continuous representation first, then quantize for TSU compatibility
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
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import time
import os
import pickle

from thermolm_jax.models.hybrid_edlm import HybridEDLM, HybridEDLMConfig
from thermolm_jax.data.wikitext_jax import WikiTextDatasetJAX
from thermolm_jax.training.checkpoint import CheckpointManager
from thermolm_jax.training.logging import WandBLogger


@dataclass
class HybridTrainingConfig:
    """Configuration for two-stage hybrid training."""
    # Model config
    vocab_size: int = 50257
    d_model: int = 512
    d_latent: int = 64
    n_levels: int = 8
    max_seq_len: int = 128
    
    # Encoder config
    num_encoder_layers: int = 6
    num_encoder_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    
    # Energy function config
    num_energy_layers: int = 6
    num_energy_heads: int = 8
    continuous_weight: float = 0.5
    discrete_weight: float = 0.5
    
    # Quantization config
    quantization_type: str = "fsq"
    commitment_cost: float = 0.25
    
    # Training config
    batch_size: int = 32
    stage1_learning_rate: float = 1e-4
    stage2_learning_rate: float = 5e-5
    weight_decay: float = 0.01
    stage1_epochs: int = 30
    stage2_epochs: int = 20
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Data config
    train_stride: int = 64
    val_stride: int = 128
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/hybrid_edlm"
    stage1_checkpoint_interval: int = 5
    stage2_checkpoint_interval: int = 5
    
    # Logging
    log_interval: int = 100
    use_wandb: bool = True
    wandb_project: str = "thermolm-jax"
    wandb_run_name: Optional[str] = None
    
    # Misc
    seed: int = 42


class HybridEDLMTrainer:
    """Trainer for hybrid EDLM model with two-stage training."""
    
    def __init__(
        self,
        config: HybridTrainingConfig,
    ):
        """Initialize trainer."""
        self.config = config
        self.key = jax.random.PRNGKey(config.seed)
        
        # Set up logging
        if config.use_wandb:
            self.logger = WandBLogger(
                project=config.wandb_project,
                run_name=config.wandb_run_name,
                config=config.__dict__,
            )
        else:
            self.logger = None
        
        # Set up checkpointing
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        
        # Initialize model
        self.model_config = HybridEDLMConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            d_latent=config.d_latent,
            n_levels=config.n_levels,
            max_seq_len=config.max_seq_len,
            num_encoder_layers=config.num_encoder_layers,
            num_encoder_heads=config.num_encoder_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            quantization_type=config.quantization_type,
            commitment_cost=config.commitment_cost,
            num_energy_layers=config.num_energy_layers,
            num_energy_heads=config.num_energy_heads,
            continuous_weight=config.continuous_weight,
            discrete_weight=config.discrete_weight,
        )
        
        self.model = HybridEDLM(self.model_config)
        
        # Load data
        self._load_data()
        
        # Initialize model parameters
        self._init_model()
    
    def _load_data(self):
        """Load training and validation data."""
        print("Loading data...")
        
        self.train_dataset = WikiTextDatasetJAX(
            split='train',
            max_length=self.config.max_seq_len,
            stride=self.config.train_stride,
        )
        
        self.val_dataset = WikiTextDatasetJAX(
            split='validation',
            max_length=self.config.max_seq_len,
            stride=self.config.val_stride,
        )
        
        # Create batches
        self.train_batches = self._create_batches(
            self.train_dataset,
            self.config.batch_size,
        )
        
        self.val_batches = self._create_batches(
            self.val_dataset,
            self.config.batch_size,
        )
        
        print(f"Train batches: {len(self.train_batches)}")
        print(f"Val batches: {len(self.val_batches)}")
    
    def _create_batches(
        self,
        dataset: WikiTextDatasetJAX,
        batch_size: int,
    ) -> jnp.ndarray:
        """Create batches from dataset."""
        examples = dataset.examples
        n_examples = len(examples)
        n_batches = n_examples // batch_size
        
        # Pad if necessary
        if n_examples % batch_size != 0:
            pad_size = batch_size - (n_examples % batch_size)
            examples = jnp.concatenate([
                examples,
                examples[:pad_size],
            ], axis=0)
            n_batches += 1
        
        # Reshape into batches
        batches = examples.reshape(n_batches, batch_size, -1)
        
        return batches
    
    def _init_model(self):
        """Initialize model parameters."""
        print("Initializing model...")
        
        # Create dummy batch for initialization
        dummy_batch = self.train_batches[0]
        
        # Initialize model
        self.key, init_key = jax.random.split(self.key)
        self.params = self.model.init(init_key, dummy_batch)
        
        # Count parameters
        param_count = sum(
            p.size for p in jax.tree_util.tree_leaves(self.params)
        )
        print(f"Model parameters: {param_count:,}")
    
    def stage1_train_step(
        self,
        params: Dict[str, Any],
        opt_state: optax.OptState,
        batch: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> Tuple[Dict[str, Any], optax.OptState, float, jax.random.PRNGKey]:
        """
        Single training step for Stage 1: continuous encoder + continuous energy.

        Args:
            params: Model parameters
            opt_state: Optimizer state
            batch: Training batch
            key: PRNG key

        Returns:
            params: Updated parameters
            opt_state: Updated optimizer state
            loss: Training loss
            key: Updated PRNG key
        """
        # Encode tokens to continuous latents (no quantization)
        latents, _, _, _ = self.model.apply(params, batch, quantize=False)

        # Compute reconstruction loss using model's compute_loss method
        # Stage 1 uses reconstruction loss to train the encoder
        loss, info = self.model.apply(
            params,
            latents=latents,
            codes=jnp.zeros_like(latents, dtype=jnp.int32),  # Dummy codes for Stage 1
            quantization_info={'commitment_loss': 0.0},
            mask=jnp.ones((batch.shape[0], batch.shape[1])),
            key=key,
            stage=1,
            tokens=batch,
            method=self.model.compute_loss,
        )

        # Compute gradients
        loss_fn = lambda p: self.model.apply(
            p,
            latents=latents,
            codes=jnp.zeros_like(latents, dtype=jnp.int32),
            quantization_info={'commitment_loss': 0.0},
            mask=jnp.ones((batch.shape[0], batch.shape[1])),
            key=key,
            stage=1,
            tokens=batch,
            method=self.model.compute_loss,
        )[0]
        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Clip gradients
        grads = jax.tree_util.tree_map(
            lambda g: jnp.clip(g, -self.config.max_grad_norm, self.config.max_grad_norm),
            grads,
        )

        # Update parameters
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss, key
    
    def stage2_train_step(
        self,
        params: Dict[str, Any],
        opt_state: optax.OptState,
        batch: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> Tuple[Dict[str, Any], optax.OptState, float, jax.random.PRNGKey, Dict[str, Any]]:
        """
        Stage 2 training step: quantization + discrete energy.
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            batch: Training batch
            key: PRNG key
        
        Returns:
            params: Updated parameters
            opt_state: Updated optimizer state
            loss: Training loss
            key: Updated PRNG key
            info: Dictionary with loss components
        """
        # Encode tokens to continuous latents and quantize
        latents, codes, quantized, quantization_info = self.model.apply(params, batch, quantize=True)
        
        # Compute hybrid energy loss
        mask = jnp.ones((batch.shape[0], batch.shape[1]))
        loss, info = self.model.apply(
            params,
            latents,
            codes,
            quantization_info,
            mask,
            key,
            method=self.model.compute_loss,
        )
        
        # Compute gradients
        loss_fn = lambda p: self.model.apply(
            p,
            latents,
            codes,
            quantization_info,
            mask,
            key,
            method=self.model.compute_loss,
        )[0]
        loss, grads = jax.value_and_grad(loss_fn)(params)
        
        # Clip gradients
        grads = jax.tree_util.tree_map(
            lambda g: jnp.clip(g, -self.config.max_grad_norm, self.config.max_grad_norm),
            grads,
        )
        
        # Update parameters
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss, key, info
    
    def train_stage1(self):
        """Train Stage 1: continuous encoder + continuous energy."""
        print("Starting Stage 1 training (continuous encoder + continuous energy)...")
        
        # Set up optimizer for Stage 1
        self.optimizer = optax.adamw(
            learning_rate=self.config.stage1_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Initialize optimizer state
        self.opt_state = self.optimizer.init(self.params)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.stage1_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_losses = []
            key = self.key
            
            for i, batch in enumerate(self.train_batches):
                self.params, self.opt_state, loss, key = self.stage1_train_step(
                    self.params,
                    self.opt_state,
                    batch,
                    key,
                )
                
                train_losses.append(loss)
                
                # Log progress
                if (i + 1) % self.config.log_interval == 0:
                    avg_loss = jnp.mean(jnp.array(train_losses[-self.config.log_interval:]))
                    print(f"Stage 1 Epoch {epoch}, Step {i+1}/{len(self.train_batches)}, Loss: {avg_loss:.4f}")
                    
                    if self.logger:
                        self.logger.log({
                            'stage': 1,
                            'epoch': epoch,
                            'step': i + 1,
                            'train_loss': float(avg_loss),
                        })

            # Compute epoch training loss
            train_loss = jnp.mean(jnp.array(train_losses))

            # Validation
            val_losses = []
            for val_batch in self.val_batches:
                # Encode tokens to continuous latents (no quantization)
                latents, _, _, _ = self.model.apply(self.params, val_batch, quantize=False)

                # Compute reconstruction loss using model's compute_loss method
                val_loss, _ = self.model.apply(
                    self.params,
                    latents=latents,
                    codes=jnp.zeros_like(latents, dtype=jnp.int32),
                    quantization_info={'commitment_loss': 0.0},
                    mask=jnp.ones((val_batch.shape[0], val_batch.shape[1])),
                    key=key,
                    stage=1,
                    tokens=val_batch,
                    method=self.model.compute_loss,
                )
                val_losses.append(val_loss)

            val_loss = jnp.mean(jnp.array(val_losses))

            # Log epoch metrics
            epoch_time = time.time() - epoch_start_time
            print(f"Stage 1 Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Time = {epoch_time:.2f}s")
            
            if self.logger:
                self.logger.log({
                    'stage': 1,
                    'epoch': epoch,
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                    'epoch_time': epoch_time,
                })
            
            # Save checkpoint
            if (epoch + 1) % self.config.stage1_checkpoint_interval == 0:
                checkpoint = {
                    'stage': 1,
                    'epoch': epoch + 1,
                    'params': self.params,
                    'opt_state': self.opt_state,
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                }
                self.checkpoint_manager.save(checkpoint, f"stage1_epoch_{epoch+1}")
                print(f"Saved Stage 1 checkpoint at epoch {epoch+1}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'stage': 1,
                    'epoch': epoch + 1,
                    'params': self.params,
                    'opt_state': self.opt_state,
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                }
                self.checkpoint_manager.save(checkpoint, "stage1_best")
                print(f"Saved Stage 1 best model with val_loss {val_loss:.4f}")
        
        print("Stage 1 training complete!")
        
        # Save final Stage 1 checkpoint
        checkpoint = {
            'stage': 1,
            'epoch': self.config.stage1_epochs,
            'params': self.params,
            'opt_state': self.opt_state,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
        }
        self.checkpoint_manager.save(checkpoint, "stage1_final")
        
        return self.params
    
    def train_stage2(self, stage1_params: Dict[str, Any]):
        """Train Stage 2: quantization + discrete energy."""
        print("Starting Stage 2 training (quantization + discrete energy)...")
        
        # Load Stage 1 parameters
        self.params = stage1_params
        
        # Freeze encoder parameters (optional - can also fine-tune)
        # For now, we'll train all parameters but with lower learning rate
        
        # Set up optimizer for Stage 2
        self.optimizer = optax.adamw(
            learning_rate=self.config.stage2_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Initialize optimizer state
        self.opt_state = self.optimizer.init(self.params)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.stage2_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_losses = []
            key = self.key
            
            for i, batch in enumerate(self.train_batches):
                self.params, self.opt_state, loss, key, info = self.stage2_train_step(
                    self.params,
                    self.opt_state,
                    batch,
                    key,
                )
                
                train_losses.append(loss)
                
                # Log progress
                if (i + 1) % self.config.log_interval == 0:
                    avg_loss = jnp.mean(jnp.array(train_losses[-self.config.log_interval:]))
                    print(f"Stage 2 Epoch {epoch}, Step {i+1}/{len(self.train_batches)}, Loss: {avg_loss:.4f}")
                    
                    if self.logger:
                        self.logger.log({
                            'stage': 2,
                            'epoch': epoch,
                            'step': i + 1,
                            'train_loss': float(avg_loss),
                        })

            # Compute epoch training loss
            train_loss = jnp.mean(jnp.array(train_losses))

            # Validation
            val_losses = []
            for val_batch in self.val_batches:
                # Encode and quantize tokens
                latents, codes, _, quantization_info = self.model.apply(self.params, val_batch, quantize=True)

                # Compute hybrid energy loss
                val_loss, _ = self.model.apply(
                    self.params,
                    latents=latents,
                    codes=codes,
                    quantization_info=quantization_info,
                    mask=jnp.ones((val_batch.shape[0], val_batch.shape[1])),
                    key=key,
                    stage=2,
                    tokens=val_batch,
                    method=self.model.compute_loss,
                )
                val_losses.append(val_loss)

            val_loss = jnp.mean(jnp.array(val_losses))

            # Log epoch metrics
            epoch_time = time.time() - epoch_start_time
            print(f"Stage 2 Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Time = {epoch_time:.2f}s")
            
            if self.logger:
                self.logger.log({
                    'stage': 2,
                    'epoch': epoch,
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                    'epoch_time': epoch_time,
                })
            
            # Save checkpoint
            if (epoch + 1) % self.config.stage2_checkpoint_interval == 0:
                checkpoint = {
                    'stage': 2,
                    'epoch': epoch + 1,
                    'params': self.params,
                    'opt_state': self.opt_state,
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                }
                self.checkpoint_manager.save(checkpoint, f"stage2_epoch_{epoch+1}")
                print(f"Saved Stage 2 checkpoint at epoch {epoch+1}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'stage': 2,
                    'epoch': epoch + 1,
                    'params': self.params,
                    'opt_state': self.opt_state,
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                }
                self.checkpoint_manager.save(checkpoint, "stage2_best")
                print(f"Saved Stage 2 best model with val_loss {val_loss:.4f}")
        
        print("Stage 2 training complete!")
        
        # Save final Stage 2 checkpoint
        checkpoint = {
            'stage': 2,
            'epoch': self.config.stage2_epochs,
            'params': self.params,
            'opt_state': self.opt_state,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
        }
        self.checkpoint_manager.save(checkpoint, "stage2_final")
        
        return self.params
    
    def train(self):
        """Full two-stage training loop."""
        print("Starting two-stage hybrid training...")
        
        # Stage 1: Train continuous encoder + continuous energy
        stage1_params = self.train_stage1()
        
        # Stage 2: Train quantization + discrete energy
        stage2_params = self.train_stage2(stage1_params)
        
        print("Two-stage hybrid training complete!")
        
        return stage2_params


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train hybrid EDLM model")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--stage1_lr', type=float, default=1e-4)
    parser.add_argument('--stage2_lr', type=float, default=5e-5)
    parser.add_argument('--stage1_epochs', type=int, default=30)
    parser.add_argument('--stage2_epochs', type=int, default=20)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_latent', type=int, default=64)
    parser.add_argument('--n_levels', type=int, default=8)
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints/hybrid_edlm")
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    
    args = parser.parse_args()
    
    config = HybridTrainingConfig(
        batch_size=args.batch_size,
        stage1_learning_rate=args.stage1_lr,
        stage2_learning_rate=args.stage2_lr,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        d_model=args.d_model,
        d_latent=args.d_latent,
        n_levels=args.n_levels,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
        wandb_run_name=args.wandb_run_name,
    )
    
    trainer = HybridEDLMTrainer(config)
    params = trainer.train()
    
    print("Hybrid training finished successfully!")


if __name__ == "__main__":
    main()
