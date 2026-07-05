"""
Training script for discrete EDLM model.

Implements training loop for discrete Energy-Based Diffusion Language Model
with FSQ quantization, discrete energy function, and THRML sampling.

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

from thermolm_jax.models.discrete_edlm import DiscreteEDLM, DiscreteEDLMConfig
from thermolm_jax.models.discrete_energy import DiscreteEnergyLoss, DiscreteEnergyConfig
from thermolm_jax.data.wikitext_jax import WikiTextDatasetJAX
from thermolm_jax.training.checkpoint import CheckpointManager
from thermolm_jax.training.logging import WandBLogger
from thermolm_jax.models.diffusion_schedule import DiffusionSchedule


@dataclass
class TrainingConfig:
    """Configuration for training discrete EDLM."""
    # Model config
    vocab_size: int = 50257
    d_model: int = 512
    d_latent: int = 64
    n_levels: int = 8
    max_seq_len: int = 128
    
    # Energy function config
    num_energy_layers: int = 6
    num_energy_heads: int = 8
    dropout: float = 0.1
    num_diffusion_timesteps: int = 1000  # Number of diffusion timesteps
    
    # Sampling config
    block_size: int = 16
    n_samples: int = 10
    n_steps: int = 100
    temperature: float = 1.0
    
    # Training config
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Diffusion config
    diffusion_schedule_type: str = "cosine"  # "cosine", "linear", "sigmoid"
    
    # Data config
    train_stride: int = 64
    val_stride: int = 128
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/discrete_edlm"
    checkpoint_interval: int = 5
    
    # Logging
    log_interval: int = 100
    use_wandb: bool = True
    wandb_project: str = "thermolm-jax"
    wandb_run_name: Optional[str] = None
    
    # Misc
    seed: int = 42
    mixed_precision: bool = False


class DiscreteEDLMTrainer:
    """Trainer for discrete EDLM model."""
    
    def __init__(
        self,
        config: TrainingConfig,
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
        self.model_config = DiscreteEDLMConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            d_latent=config.d_latent,
            n_levels=config.n_levels,
            max_seq_len=config.max_seq_len,
            num_energy_layers=config.num_energy_layers,
            num_energy_heads=config.num_energy_heads,
            dropout=config.dropout,
            block_size=config.block_size,
            n_samples=config.n_samples,
            n_steps=config.n_steps,
            temperature=config.temperature,
        )
        
        self.model = DiscreteEDLM(self.model_config)
        
        # Initialize loss function
        self.loss_config = DiscreteEnergyConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            d_latent=config.d_latent,
            n_levels=config.n_levels,
            num_energy_layers=config.num_energy_layers,
            num_energy_heads=config.num_energy_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            num_diffusion_timesteps=config.num_diffusion_timesteps,
        )
        
        self.loss_fn = DiscreteEnergyLoss(self.loss_config)
        
        # Initialize diffusion schedule
        self.diffusion_schedule = DiffusionSchedule(
            schedule_type=config.diffusion_schedule_type,
            T=config.num_diffusion_timesteps,
        )
        
        # Set up optimizer
        self.optimizer = optax.adamw(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
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
        
        # Initialize loss function parameters
        self.key, loss_key = jax.random.split(self.key)
        dummy_codes = jax.random.randint(
            loss_key,
            shape=(self.config.batch_size, self.config.max_seq_len, self.config.d_latent),
            minval=0,
            maxval=self.config.n_levels,
        )
        self.loss_params = self.loss_fn.init(loss_key, dummy_codes)
        
        # Initialize optimizer state
        self.opt_state = self.optimizer.init(self.params)
        
        # Count parameters
        param_count = sum(
            p.size for p in jax.tree_util.tree_leaves(self.params)
        )
        print(f"Model parameters: {param_count:,}")
    
    def train_step(
        self,
        params: Dict[str, Any],
        opt_state: optax.OptState,
        batch: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> Tuple[Dict[str, Any], optax.OptState, float, jax.random.PRNGKey]:
        """
        Single training step with diffusion timestep sampling.
        
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
        # Encode tokens to discrete codes
        key, encode_key = jax.random.split(key)
        codes, latents, _ = self.model.apply(params, batch)
        
        # Sample random timesteps for diffusion
        key, t_key, loss_key = jax.random.split(key, 3)
        t = jax.random.randint(
            t_key,
            shape=(batch.shape[0],),
            minval=0,
            maxval=self.config.num_diffusion_timesteps,
        )
        
        # Compute loss with timestep information
        mask = jnp.ones((batch.shape[0], batch.shape[1]))
        loss, grads = jax.value_and_grad(self.loss_fn.apply)(
            self.loss_params,
            codes,
            mask=mask,
            key=loss_key,
        )
        
        # Clip gradients
        grads = jax.tree_util.tree_map(
            lambda g: jnp.clip(g, -self.config.max_grad_norm, self.config.max_grad_norm),
            grads,
        )
        
        # Update parameters
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss, key
    
    def validate(
        self,
        params: Dict[str, Any],
        val_batches: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> float:
        """
        Validate model on validation set.
        
        Args:
            params: Model parameters
            val_batches: Validation batches
            key: PRNG key
        
        Returns:
            val_loss: Validation loss
        """
        val_losses = []
        
        for batch in val_batches:
            # Encode tokens to discrete codes
            key, encode_key = jax.random.split(key)
            codes, latents, _ = self.model.apply(params, batch)
            
            # Compute loss
            key, loss_key = jax.random.split(key)
            mask = jnp.ones((batch.shape[0], batch.shape[1]))
            loss = self.loss_fn.apply(
                self.loss_params,
                codes,
                mask=mask,
                key=loss_key,
            )
            
            val_losses.append(loss)
        
        val_loss = jnp.mean(jnp.array(val_losses))
        
        return val_loss
    
    def train(self):
        """Full training loop."""
        print("Starting training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_losses = []
            key = self.key
            
            for i, batch in enumerate(self.train_batches):
                self.params, self.opt_state, loss, key = self.train_step(
                    self.params,
                    self.opt_state,
                    batch,
                    key,
                )
                
                train_losses.append(loss)
                
                # Log progress
                if (i + 1) % self.config.log_interval == 0:
                    avg_loss = jnp.mean(jnp.array(train_losses[-self.config.log_interval:]))
                    print(f"Epoch {epoch}, Step {i+1}/{len(self.train_batches)}, Loss: {avg_loss:.4f}")
                    
                    if self.logger:
                        self.logger.log({
                            'epoch': epoch,
                            'step': i + 1,
                            'train_loss': float(avg_loss),
                        })
            
            # Compute epoch training loss
            train_loss = jnp.mean(jnp.array(train_losses))
            
            # Validation
            val_loss = self.validate(self.params, self.val_batches, key)
            
            # Log epoch metrics
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Time = {epoch_time:.2f}s")
            
            if self.logger:
                self.logger.log({
                    'epoch': epoch,
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                    'epoch_time': epoch_time,
                })
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'params': self.params,
                    'opt_state': self.opt_state,
                    'loss_params': self.loss_params,
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                }
                self.checkpoint_manager.save(checkpoint, f"epoch_{epoch+1}")
                print(f"Saved checkpoint at epoch {epoch+1}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch + 1,
                    'params': self.params,
                    'opt_state': self.opt_state,
                    'loss_params': self.loss_params,
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                }
                self.checkpoint_manager.save(checkpoint, "best")
                print(f"Saved best model with val_loss {val_loss:.4f}")
        
        print("Training complete!")
        
        # Save final checkpoint
        checkpoint = {
            'epoch': self.config.num_epochs,
            'params': self.params,
            'opt_state': self.opt_state,
            'loss_params': self.loss_params,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
        }
        self.checkpoint_manager.save(checkpoint, "final")
        
        return self.params


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train discrete EDLM model")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_latent', type=int, default=64)
    parser.add_argument('--n_levels', type=int, default=8)
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints/discrete_edlm")
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        d_model=args.d_model,
        d_latent=args.d_latent,
        n_levels=args.n_levels,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
        wandb_run_name=args.wandb_run_name,
    )
    
    trainer = DiscreteEDLMTrainer(config)
    params = trainer.train()
    
    print("Training finished successfully!")


if __name__ == "__main__":
    main()
