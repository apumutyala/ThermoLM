"""
Preprocessing utilities for ThermoLM JAX.

Provides common preprocessing functions for text data.

Author: Apuroop Mutyala
Date: April 14, 2026
"""

import jax.numpy as jnp
from typing import List, Tuple, Optional


def pad_sequence(
    tokens: List[int],
    max_length: int,
    pad_token_id: int,
    eos_token_id: Optional[int] = None,
) -> jnp.ndarray:
    """
    Pad sequence to max_length.
    
    Args:
        tokens: Input token IDs
        max_length: Target length
        pad_token_id: Padding token ID
        eos_token_id: If provided, force last token to be EOS
    
    Returns:
        padded: Padded sequence as JAX array
    """
    if len(tokens) >= max_length:
        # Truncate if too long
        tokens = tokens[:max_length]
        if eos_token_id is not None:
            tokens[-1] = eos_token_id
        return jnp.array(tokens, dtype=jnp.int32)
    
    # Pad if needed
    padding = [pad_token_id] * (max_length - len(tokens))
    padded = tokens + padding
    
    # Force EOS at end if specified
    if eos_token_id is not None:
        padded[-1] = eos_token_id
    
    return jnp.array(padded, dtype=jnp.int32)


def truncate_sequence(
    tokens: List[int],
    max_length: int,
    eos_token_id: Optional[int] = None,
) -> List[int]:
    """
    Truncate sequence to max_length.
    
    Args:
        tokens: Input token IDs
        max_length: Target length
        eos_token_id: If provided, force last token to be EOS
    
    Returns:
        truncated: Truncated sequence
    """
    if len(tokens) <= max_length:
        if eos_token_id is not None:
            tokens = tokens.copy()
            tokens[-1] = eos_token_id
        return tokens
    
    # Truncate
    truncated = tokens[:max_length]
    
    # Force EOS at end if specified
    if eos_token_id is not None:
        truncated[-1] = eos_token_id
    
    return truncated


def create_sliding_windows(
    tokens: List[int],
    max_length: int,
    stride: int,
    min_length: Optional[int] = None,
) -> List[List[int]]:
    """
    Create sliding windows from tokens.
    
    Args:
        tokens: Input token IDs
        max_length: Window length
        stride: Stride between windows
        min_length: Minimum window length (default: max_length // 2)
    
    Returns:
        windows: List of token windows
    """
    if min_length is None:
        min_length = max_length // 2
    
    windows = []
    for i in range(0, len(tokens), stride):
        window = tokens[i:i + max_length]
        if len(window) >= min_length:
            windows.append(window)
    
    return windows


def batch_sequences(
    sequences: List[jnp.ndarray],
    batch_size: int,
    drop_last: bool = True,
) -> jnp.ndarray:
    """
    Batch sequences into JAX array.
    
    Args:
        sequences: List of sequences
        batch_size: Batch size
        drop_last: Whether to drop last incomplete batch
    
    Returns:
        batches: (n_batches, batch_size, seq_len) array
    """
    n_sequences = len(sequences)
    n_batches = n_sequences // batch_size
    
    if drop_last:
        sequences = sequences[:n_batches * batch_size]
    else:
        # Pad last batch if needed
        remainder = n_sequences % batch_size
        if remainder > 0:
            pad = [sequences[-1]] * (batch_size - remainder)
            sequences = sequences + pad
            n_batches += 1
    
    # Stack into batches
    batches = []
    for i in range(n_batches):
        batch = sequences[i * batch_size:(i + 1) * batch_size]
        batches.append(jnp.stack(batch))
    
    return jnp.stack(batches)


def mask_sequence(
    tokens: jnp.ndarray,
    mask_token_id: int,
    mask_prob: float = 0.15,
    key: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Randomly mask tokens (for masked language modeling).
    
    Args:
        tokens: Input token IDs
        mask_token_id: Mask token ID
        mask_prob: Probability of masking each token
        key: PRNG key for randomness
    
    Returns:
        masked: Masked token IDs
        mask: Boolean mask (True where masked)
    """
    import jax.random as random
    
    if key is None:
        raise ValueError("PRNG key must be provided for mask generation")
    
    # Generate random mask
    mask = random.uniform(key, tokens.shape) < mask_prob
    
    # Apply mask
    masked = jnp.where(mask, mask_token_id, tokens)
    
    return masked, mask
