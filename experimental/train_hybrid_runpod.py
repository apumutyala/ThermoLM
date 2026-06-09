"""
RunPod training script for hybrid EDLM model.

This script is designed to run on RunPod with proper environment setup.
Handles two-stage training for hybrid continuous-discrete model.

Author: Apuroop Mutyala
Date: April 15, 2026
"""

import os
import subprocess
import sys

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    commands = [
        "pip install --upgrade pip",
        "pip install jax jaxlib[cuda12_local] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        "pip install flax optax",
        "pip install transformers datasets",
        "pip install wandb",
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
    
    print("Dependencies installed successfully!")


def setup_environment():
    """Set up environment variables."""
    os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    print("Environment set up successfully!")


def main():
    """Main function."""
    print("Setting up RunPod environment for hybrid EDLM training...")
    
    # Install dependencies
    install_dependencies()
    
    # Set up environment
    setup_environment()
    
    # Run training
    print("Starting two-stage hybrid training...")
    from thermolm_jax.training.train_hybrid_edlm import main as train_main
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Train hybrid EDLM on RunPod")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--stage1_lr', type=float, default=1e-4, help='Stage 1 learning rate')
    parser.add_argument('--stage2_lr', type=float, default=5e-5, help='Stage 2 learning rate')
    parser.add_argument('--stage1_epochs', type=int, default=30, help='Stage 1 epochs')
    parser.add_argument('--stage2_epochs', type=int, default=20, help='Stage 2 epochs')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--d_latent', type=int, default=64, help='Latent dimension')
    parser.add_argument('--n_levels', type=int, default=8, help='Number of quantization levels')
    parser.add_argument('--checkpoint_dir', type=str, default="/output/checkpoints/hybrid_edlm", help='Checkpoint directory')
    parser.add_argument('--use_wandb', action='store_true', default=True, help='Use Weights & Biases logging')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
    
    args = parser.parse_args()
    
    # Update sys.argv for the training script
    sys.argv = [
        'train_hybrid_edlm.py',
        '--batch_size', str(args.batch_size),
        '--stage1_lr', str(args.stage1_lr),
        '--stage2_lr', str(args.stage2_lr),
        '--stage1_epochs', str(args.stage1_epochs),
        '--stage2_epochs', str(args.stage2_epochs),
        '--d_model', str(args.d_model),
        '--d_latent', str(args.d_latent),
        '--n_levels', str(args.n_levels),
        '--checkpoint_dir', args.checkpoint_dir,
    ]
    
    if args.use_wandb:
        sys.argv.append('--use_wandb')
    
    if args.wandb_run_name:
        sys.argv.extend(['--wandb_run_name', args.wandb_run_name])
    
    # Run training
    train_main()


if __name__ == "__main__":
    main()
