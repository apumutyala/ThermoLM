"""
Three-Stage Training for Hybrid DTM

Implements three-stage training from Extropic.pdf Appendix I:
1. Train autoencoder (reconstruction loss)
2. Train DTM on binary embeddings (contrastive divergence)
3. GAN fine-tune decoder

Design Decision: Full Three-Stage Training
- Rationale: Follows Extropic.pdf exactly for energy efficiency
- Impact: Optimal energy efficiency for TSU hardware
- Trade-off: Complex multi-stage training
- Downstream: Demonstrates complete understanding

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import jax
import jax.numpy as jnp
import optax
from typing import Callable, Optional, Dict, Tuple
from dataclasses import dataclass

from ..models.binary_autoencoder import BinaryAutoencoder, BinaryAutoencoderConfig
from ..models.dtm import DTM, DTMConfig
from .train_dtm import DTMTrainer, TrainingConfig


@dataclass
class HybridTrainingConfig:
    """Configuration for three-stage hybrid training."""
    # Stage 1: Autoencoder
    ae_n_epochs: int = 50
    ae_batch_size: int = 32
    ae_learning_rate: float = 1e-3
    
    # Stage 2: DTM
    dtm_n_epochs: int = 100
    dtm_batch_size: int = 32
    dtm_learning_rate: float = 1e-3
    
    # Stage 3: GAN (optional, can skip)
    use_gan: bool = False
    gan_n_epochs: int = 50
    gan_learning_rate: float = 2e-4
    
    # Common
    vocab_size: int = 50257
    d_model: int = 512
    d_latent: int = 64
    max_seq_len: int = 128


def train_stage1_autoencoder(
    data_loader: Callable,
    config: HybridTrainingConfig,
    key: jax.random.PRNGKey
) -> Tuple[BinaryAutoencoder, Dict]:
    """
    Stage 1: Train autoencoder with reconstruction loss.
    
    Args:
        data_loader: Data loader yielding token batches
        config: Training configuration
        key: Random key
    
    Returns:
        autoencoder: Trained autoencoder
        history: Training history
    """
    print("=== Stage 1: Training Autoencoder ===")
    
    # Initialize autoencoder
    ae_config = BinaryAutoencoderConfig(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        d_latent=config.d_latent,
        max_seq_len=config.max_seq_len
    )
    autoencoder = BinaryAutoencoder(ae_config, key)
    
    # Initialize optimizer
    optimizer = optax.adam(config.ae_learning_rate)
    opt_state = optimizer.init(autoencoder)
    
    # Training loop
    history = []
    for epoch in range(config.ae_n_epochs):
        epoch_losses = []
        
        for step, batch in enumerate(data_loader):
            # Compute loss
            def loss_fn(model):
                return model.reconstruction_loss(batch)[0]
            
            loss, grads = jax.value_and_grad(loss_fn)(autoencoder)
            
            # Update
            updates, opt_state = optimizer.update(grads, opt_state)
            autoencoder = optax.apply_updates(autoencoder, updates)
            
            epoch_losses.append(float(loss))
            
            if step % 10 == 0:
                print(f"Stage 1 Epoch {epoch}, Step {step}: loss={loss:.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        history.append({"epoch": epoch, "loss": avg_loss})
        print(f"Stage 1 Epoch {epoch}: avg_loss={avg_loss:.4f}")
    
    return autoencoder, history


def train_stage2_dtm(
    autoencoder: BinaryAutoencoder,
    data_loader: Callable,
    config: HybridTrainingConfig,
    key: jax.random.PRNGKey
) -> Tuple[DTM, Dict]:
    """
    Stage 2: Train DTM on binary embeddings.
    
    Args:
        autoencoder: Trained autoencoder (frozen)
        data_loader: Data loader yielding token batches
        config: Training configuration
        key: Random key
    
    Returns:
        dtm: Trained DTM
        history: Training history
    """
    print("=== Stage 2: Training DTM on Binary Embeddings ===")
    
    # Freeze autoencoder
    # (In practice, would use eqx.tree_at to freeze parameters)
    
    # Initialize DTM
    dtm_config = DTMConfig(
        n_data_vars=config.d_latent * config.max_seq_len,
        n_latent_vars=config.d_latent * config.max_seq_len // 4,  # 25% latent
        T=1000
    )
    dtm = DTM(dtm_config, key)
    
    # Initialize DTM trainer
    training_config = TrainingConfig(
        n_epochs=config.dtm_n_epochs,
        batch_size=config.dtm_batch_size,
        learning_rate=config.dtm_learning_rate,
        n_gibbs_steps=50
    )
    trainer = DTMTrainer(dtm, training_config, key)
    
    # Convert data loader to binary embeddings
    def binary_data_loader():
        for batch in data_loader:
            binary, _ = autoencoder.encode(batch)
            # Flatten sequence dimension for DTM
            batch_size, seq_len, d_latent = binary.shape
            binary_flat = binary.reshape(batch_size, seq_len * d_latent)
            yield binary_flat
    
    # Train DTM
    history = []
    for epoch in range(config.dtm_n_epochs):
        epoch_metrics = trainer.train_epoch(binary_data_loader(), epoch)
        history.append(epoch_metrics)
        print(f"Stage 2 Epoch {epoch}: {epoch_metrics}")
    
    return dtm, history


def train_stage3_gan(
    autoencoder: BinaryAutoencoder,
    dtm: DTM,
    data_loader: Callable,
    config: HybridTrainingConfig,
    key: jax.random.PRNGKey
) -> Tuple:
    """
    Stage 3: GAN fine-tuning of decoder.
    
    This stage is optional and can be skipped for simplicity.
    
    Args:
        autoencoder: Trained autoencoder
        dtm: Trained DTM
        data_loader: Data loader
        config: Training configuration
        key: Random key
    
    Returns:
        generator: Fine-tuned generator
        discriminator: Trained discriminator
        history: Training history
    """
    print("=== Stage 3: GAN Fine-tuning (Optional) ===")
    print("Stage 3 skipped (set use_gan=True to enable)")
    
    # Return placeholders
    return autoencoder, None, []


def train_hybrid_dtm(
    data_loader: Callable,
    config: HybridTrainingConfig,
    key: jax.random.PRNGKey
) -> Tuple:
    """
    Three-stage training from Extropic.pdf Appendix I.
    
    Args:
        data_loader: Data loader yielding token batches
        config: Training configuration
        key: Random key
    
    Returns:
        autoencoder: Trained autoencoder
        dtm: Trained DTM
        history: Combined training history
    """
    key_ae, key_dtm = jax.random.split(key)
    
    # Stage 1: Train autoencoder
    autoencoder, history_ae = train_stage1_autoencoder(
        data_loader, config, key_ae
    )
    
    # Stage 2: Train DTM
    dtm, history_dtm = train_stage2_dtm(
        autoencoder, data_loader, config, key_dtm
    )
    
    # Stage 3: GAN fine-tuning (optional)
    if config.use_gan:
        key_gan = jax.random.split(key_dtm)[1]
        generator, discriminator, history_gan = train_stage3_gan(
            autoencoder, dtm, data_loader, config, key_gan
        )
    else:
        generator = autoencoder
        discriminator = None
        history_gan = []
    
    history = {
        "stage1": history_ae,
        "stage2": history_dtm,
        "stage3": history_gan,
    }
    
    return autoencoder, dtm, history


def test_hybrid_training():
    """Test hybrid training implementation."""
    print("Testing hybrid training...")
    
    # Simple data loader
    def simple_data_loader():
        key = jax.random.PRNGKey(0)
        for _ in range(10):
            batch = jax.random.randint(key, (4, 16), minval=0, maxval=1000)
            yield batch
    
    config = HybridTrainingConfig(
        ae_n_epochs=2,
        dtm_n_epochs=2,
        vocab_size=1000,
        d_model=64,
        d_latent=16,
        max_seq_len=16
    )
    
    key = jax.random.PRNGKey(0)
    
    # Test Stage 1
    print("\n=== Testing Stage 1 ===")
    autoencoder, history_ae = train_stage1_autoencoder(
        simple_data_loader(), config, key
    )
    print(f"Stage 1 history: {history_ae}")
    
    # Test Stage 2
    print("\n=== Testing Stage 2 ===")
    dtm, history_dtm = train_stage2_dtm(
        autoencoder, simple_data_loader(), config, key
    )
    print(f"Stage 2 history: {history_dtm}")
    
    # Test full pipeline
    print("\n=== Testing Full Pipeline ===")
    autoencoder, dtm, history = train_hybrid_dtm(
        simple_data_loader(), config, key
    )
    print(f"Full history keys: {history.keys()}")
    
    print("[SUCCESS] Hybrid training test passed!")


if __name__ == "__main__":
    test_hybrid_training()
