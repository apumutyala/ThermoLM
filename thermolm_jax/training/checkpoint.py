"""
Checkpoint Module for ThermoLM JAX

Provides checkpointing utilities for training.

Design Decision: Simple pickle-based checkpointing
- Rationale: Avoid complex orbax dependency issues
- Impact: Simpler, more portable checkpointing
- Trade-off: Less features than orbax (no async, no compression)
- Downstream: Easy to load checkpoints on any machine

Author: Apuroop Mutyala
Date: April 2026
"""

import pickle
import os
from typing import Dict, Any, Optional
import jax


def save_checkpoint(
    params: Dict[str, Any],
    opt_state: Any,
    epoch: int,
    path: str,
    **kwargs
) -> None:
    """
    Save training checkpoint using pickle.
    
    Args:
        params: Model parameters
        opt_state: Optimizer state
        epoch: Current epoch
        path: Checkpoint path
        **kwargs: Additional data to save
    """
    checkpoint = {
        'params': params,
        'opt_state': opt_state,
        'epoch': epoch,
        **kwargs
    }
    
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load training checkpoint using pickle.
    
    Args:
        path: Checkpoint path
    
    Returns:
        checkpoint: Dictionary containing saved data
    """
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    return checkpoint


class CheckpointManager:
    """
    Checkpoint manager for training.
    
    Manages checkpoint saving and loading with pickle.
    """
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, checkpoint: Dict[str, Any], name: str) -> None:
        """
        Save checkpoint.
        
        Args:
            checkpoint: Checkpoint data
            name: Checkpoint name
        """
        path = os.path.join(self.checkpoint_dir, f"{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load(self, name: str) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            name: Checkpoint name
        
        Returns:
            checkpoint: Checkpoint data
        """
        path = os.path.join(self.checkpoint_dir, f"{name}.pkl")
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint
    
    def exists(self, name: str) -> bool:
        """
        Check if checkpoint exists.
        
        Args:
            name: Checkpoint name
        
        Returns:
            exists: Whether checkpoint exists
        """
        path = os.path.join(self.checkpoint_dir, f"{name}.pkl")
        return os.path.exists(path)


def create_checkpoint_manager(checkpoint_dir: str) -> CheckpointManager:
    """
    Create checkpoint manager.
    
    Args:
        checkpoint_dir: Directory for checkpoints
    
    Returns:
        manager: Checkpoint manager
    """
    return CheckpointManager(checkpoint_dir)
