"""
WikiText-2 Data Loader for ThermoLM JAX

JAX-compatible data loading pipeline for WikiText-2 dataset.
Adapted from the PyTorch version with JAX arrays.

Author: Apuroop Mutyala
Date: April 2026
"""

import jax.numpy as jnp
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from typing import List, Optional


class WikiTextDatasetJAX:
    """
    WikiText-2 dataset for ThermoLM JAX training.
    
    Features:
    - Tokenizes with GPT-2 tokenizer
    - Creates overlapping windows for better coverage
    - Handles edge cases (empty lines, very long sequences)
    - Returns JAX arrays instead of PyTorch tensors
    
    Args:
        split: 'train', 'validation', or 'test'
        max_length: Maximum sequence length (default: 128)
        stride: Stride for sliding window (default: 64)
        cache_dir: Directory to cache dataset
    """
    
    def __init__(
        self,
        split: str = 'train',
        max_length: int = 128,
        stride: int = 64,
        cache_dir: Optional[str] = None,
    ):
        self.split = split
        self.max_length = max_length
        self.stride = stride
        
        # Initialize tokenizer (GPT-2)
        print(f"Loading GPT-2 tokenizer...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load WikiText-2
        print(f"Loading WikiText-2 {split} split...")
        dataset = load_dataset(
            'wikitext',
            'wikitext-2-raw-v1',
            split=split,
            cache_dir=cache_dir,
        )
        
        # Tokenize and create windows
        print(f"Tokenizing and creating windows (max_length={max_length}, stride={stride})...")
        self.examples = self._create_examples(dataset)
        
        print(f"Created {len(self.examples)} examples from {len(dataset)} documents")
    
    def _create_examples(self, dataset) -> List[jnp.ndarray]:
        """
        Tokenize text and create overlapping windows.
        
        Key assumptions:
        - Empty lines are filtered out
        - Very short sequences (<10 tokens) are skipped
        - Overlapping windows improve coverage
        
        Returns:
            List of tokenized sequences (each max_length tokens)
        """
        examples = []
        
        for item in dataset:
            text = item['text'].strip()
            
            # Skip empty or very short lines
            if len(text) < 10:
                continue
            
            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Skip if too short
            if len(tokens) < 10:
                continue
            
            # Create windows with stride
            for i in range(0, len(tokens), self.stride):
                window = tokens[i:i + self.max_length]
                
                # Only keep if at least half of max_length
                if len(window) >= self.max_length // 2:
                    # Pad if needed
                    if len(window) < self.max_length:
                        window = window + [self.tokenizer.pad_token_id] * (self.max_length - len(window))
                    
                    # Force sequence to terminate with EOS for static block alignment
                    window[-1] = self.tokenizer.eos_token_id
                    
                    examples.append(jnp.array(window, dtype=jnp.int32))
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> jnp.ndarray:
        """
        Get a single example.
        
        Returns:
            tokens: (max_length,) token IDs
        """
        return self.examples[idx]


def create_jax_dataloaders(
    batch_size: int = 32,
    max_length: int = 128,
    stride: int = 64,
    cache_dir: Optional[str] = None,
) -> dict:
    """
    Create train and validation dataloaders for WikiText-2.
    
    JAX doesn't use DataLoader like PyTorch. Instead, we batch
    the dataset into JAX arrays for efficient GPU/TPU processing.
    
    Args:
        batch_size: Batch size
        max_length: Maximum sequence length
        stride: Stride for overlapping windows
        cache_dir: Cache directory for dataset
    
    Returns:
        dict with 'train' and 'valid' batched JAX arrays
    """
    
    # Create datasets
    train_dataset = WikiTextDatasetJAX(
        split='train',
        max_length=max_length,
        stride=stride,
        cache_dir=cache_dir,
    )
    
    valid_dataset = WikiTextDatasetJAX(
        split='validation',
        max_length=max_length,
        stride=max_length,  # No overlap for validation
        cache_dir=cache_dir,
    )
    
    # Batch datasets into JAX arrays
    def batch_dataset(dataset, batch_size):
        n_examples = len(dataset)
        n_batches = n_examples // batch_size
        batches = []
        for i in range(n_batches):
            batch = dataset[i * batch_size : (i + 1) * batch_size]
            batch = jnp.stack(batch)
            batches.append(batch)
        return jnp.stack(batches)
    
    train_batches = batch_dataset(train_dataset, batch_size)
    valid_batches = batch_dataset(valid_dataset, batch_size)
    
    print(f"\nDataLoader stats:")
    print(f"  Train batches: {train_batches.shape[0]}")
    print(f"  Valid batches: {valid_batches.shape[0]}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {max_length}")
    
    return {
        'train': train_batches,
        'valid': valid_batches,
    }


if __name__ == '__main__':
    """Test data loading."""
    print("Testing WikiText-2 JAX data loader...")
    
    # Create dataloaders (small for testing)
    loaders = create_jax_dataloaders(
        batch_size=4,
        max_length=128,
        stride=64,
    )
    
    # Test train loader
    print("\nTesting train loader...")
    train_batch = loaders['train'][0]
    print(f"  Batch shape: {train_batch.shape}")
    print(f"  Batch dtype: {train_batch.dtype}")
    print(f"  Min token ID: {train_batch.min()}")
    print(f"  Max token ID: {train_batch.max()}")
    
    # Test valid loader
    print("\nTesting valid loader...")
    valid_batch = loaders['valid'][0]
    print(f"  Batch shape: {valid_batch.shape}")
    
    # Decode a sample
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    sample_text = tokenizer.decode(train_batch[0][:50])
    print(f"\nSample text (first 50 tokens):")
    print(f"  {sample_text}")
    
    print("\n✓ JAX data loader test passed!")
