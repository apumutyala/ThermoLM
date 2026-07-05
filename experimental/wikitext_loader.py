"""
WikiText-2 Data Loader for ThermoLM

GPU-optimized data loading pipeline with:
- Efficient tokenization (GPT-2)
- Smart batching with padding
- Pin memory for fast GPU transfer
- Gradient-friendly data loading

Author: ThermoLM Team
Date: April 2026
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from typing import Dict, List, Optional
import warnings


class WikiTextDataset(Dataset):
    """
    WikiText-2 dataset for ThermoLM training.
    
    Features:
    - Tokenizes with GPT-2 tokenizer
    - Creates overlapping windows for better coverage
    - Handles edge cases (empty lines, very long sequences)
    
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
        super().__init__()
        
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
    
    def _create_examples(self, dataset) -> List[torch.Tensor]:
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
                    
                    examples.append(torch.tensor(window, dtype=torch.long))
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single example.
        
        Returns:
            tokens: (max_length,) token IDs
        """
        return self.examples[idx]


def create_dataloaders(
    batch_size: int = 32,
    max_length: int = 128,
    stride: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    cache_dir: Optional[str] = None,
) -> Dict[str, DataLoader]:
    """
    Create train and validation dataloaders for WikiText-2.
    
    GPU Optimizations:
    - pin_memory=True for faster CPU→GPU transfer
    - persistent_workers for efficiency
    - prefetch_factor for pipeline parallelism
    
    Args:
        batch_size: Batch size
        max_length: Maximum sequence length
        stride: Stride for overlapping windows
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU (set True for CUDA)
        cache_dir: Cache directory for dataset
    
    Returns:
        dict with 'train' and 'valid' DataLoaders
    """
    
    # Create datasets
    train_dataset = WikiTextDataset(
        split='train',
        max_length=max_length,
        stride=stride,
        cache_dir=cache_dir,
    )
    
    valid_dataset = WikiTextDataset(
        split='validation',
        max_length=max_length,
        stride=max_length,  # No overlap for validation
        cache_dir=cache_dir,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,  # Drop last incomplete batch for stability
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )
    
    print(f"\nDataLoader stats:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Valid batches: {len(valid_loader)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {max_length}")
    print(f"  Pin memory: {pin_memory}")
    
    return {
        'train': train_loader,
        'valid': valid_loader,
    }


if __name__ == '__main__':
    """Test data loading."""
    print("Testing WikiText-2 data loader...")
    
    # Create dataloaders (small for testing)
    loaders = create_dataloaders(
        batch_size=4,
        max_length=128,
        stride=64,
        num_workers=0,  # 0 for testing
        pin_memory=False,  # False for CPU testing
    )
    
    # Test train loader
    print("\nTesting train loader...")
    train_batch = next(iter(loaders['train']))
    print(f"  Batch shape: {train_batch.shape}")
    print(f"  Batch dtype: {train_batch.dtype}")
    print(f"  Min token ID: {train_batch.min().item()}")
    print(f"  Max token ID: {train_batch.max().item()}")
    
    # Test valid loader
    print("\nTesting valid loader...")
    valid_batch = next(iter(loaders['valid']))
    print(f"  Batch shape: {valid_batch.shape}")
    
    # Decode a sample
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    sample_text = tokenizer.decode(train_batch[0][:50])
    print(f"\nSample text (first 50 tokens):")
    print(f"  {sample_text}")
    
    print("\n✓ Data loader test passed!")
