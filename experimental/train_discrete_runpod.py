"""
RunPod training script for discrete EDLM model.

This script is designed to run on RunPod with proper environment setup.
Handles data downloading, dependency installation, and training execution.

Author: Apuroop Mutyala
Date: April 15, 2026
"""

import os
import subprocess
import sys

def install_dependencies():
    """Install required dependencies with error handling."""
    print("Installing dependencies...")
    
    commands = [
        "pip install --upgrade pip",
        "pip install jax jaxlib[cuda12_local] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        "pip install flax optax",
        "pip install transformers datasets",
        "pip install wandb",
        "pip install blackjax diffusionjax",
    ]
    
    failed_commands = []
    for cmd in commands:
        print(f"Running: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"Success: {cmd}")
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {cmd}")
            print(f"Error output: {e.stderr}")
            failed_commands.append(cmd)
    
    if failed_commands:
        print(f"Warning: {len(failed_commands)} commands failed:")
        for cmd in failed_commands:
            print(f"  - {cmd}")
        print("Attempting to continue anyway...")
    else:
        print("Dependencies installed successfully!")


def download_data():
    """Download WikiText-2 dataset."""
    print("Downloading WikiText-2 dataset...")
    
    # The dataset will be downloaded automatically by HuggingFace datasets
    # when we first import WikiTextDatasetJAX
    print("Data will be downloaded automatically during training")


def setup_environment():
    """Set up environment variables with validation."""
    print("Setting up environment variables...")
    
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Set WandB API key (optional)
    wandb_key = os.getenv('WANDB_API_KEY')
    if wandb_key:
        os.environ['WANDB_API_KEY'] = wandb_key
        print("WandB API key found")
    else:
        print("Warning: WANDB_API_KEY not set, logging will be disabled")
    
    print("Environment set up successfully!")


def main():
    """Main function."""
    print("Setting up RunPod environment for discrete EDLM training...")
    
    # Install dependencies
    install_dependencies()
    
    # Download data (will happen automatically)
    download_data()
    
    # Set up environment
    setup_environment()
    
    # Run training
    print("Starting training...")
    from thermolm_jax.training.train_discrete_edlm import main as train_main
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Train discrete EDLM on RunPod")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--d_latent', type=int, default=64, help='Latent dimension')
    parser.add_argument('--n_levels', type=int, default=8, help='Number of quantization levels')
    parser.add_argument('--checkpoint_dir', type=str, default="/output/checkpoints/discrete_edlm", help='Checkpoint directory')
    parser.add_argument('--use_wandb', action='store_true', default=True, help='Use Weights & Biases logging')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
    
    args = parser.parse_args()
    
    # Update sys.argv for the training script
    sys.argv = [
        'train_discrete_edlm.py',
        '--batch_size', str(args.batch_size),
        '--learning_rate', str(args.learning_rate),
        '--num_epochs', str(args.num_epochs),
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
