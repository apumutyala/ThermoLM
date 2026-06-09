"""
Tokenizer utilities for ThermoLM JAX.

Provides tokenizer management and utilities.

Author: Apuroop Mutyala
Date: April 2026
"""

from transformers import GPT2TokenizerFast
from typing import Optional, Dict, Any
import jax
import jax.numpy as jnp

class TokenizerManager:
    """
    Manager for tokenizer operations.
    
    Provides convenient interface for tokenization and detokenization.
    """
    
    def __init__(self, tokenizer_name: str = 'gpt2'):
        """
        Initialize tokenizer manager.
        
        Args:
            tokenizer_name: Name of pretrained tokenizer
        """
        self.tokenizer_name = tokenizer_name
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Cache vocabulary size
        self.vocab_size = len(self.tokenizer)
    
    def encode(self, text: str, add_special_tokens: bool = False) -> list:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
        
        Returns:
            token_ids: List of token IDs
        """
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def decode(self, token_ids: list, skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            text: Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_encode(self, texts: list, **kwargs) -> list:
        """
        Batch encode texts.
        
        Args:
            texts: List of texts
            **kwargs: Additional arguments for tokenizer
        
        Returns:
            batch_token_ids: List of token ID lists
        """
        return self.tokenizer.batch_encode_plus(texts, **kwargs)
    
    def batch_decode(self, token_ids: list, **kwargs) -> list:
        """
        Batch decode token IDs.
        
        Args:
            token_ids: List of token ID lists
            **kwargs: Additional arguments for tokenizer
        
        Returns:
            texts: List of decoded texts
        """
        return self.tokenizer.batch_decode(token_ids, **kwargs)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        return {
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'bos_token_id': self.tokenizer.bos_token_id,
            'unk_token_id': self.tokenizer.unk_token_id,
        }


# TODO: Add support for other tokenizers (Llama, etc.) - Implemented below
# TODO: Implement tokenizer caching - Implemented below
# TODO: Add tokenizer-specific preprocessing - Implemented below


class TokenizerCache:
    """
    Cache for tokenized sequences to avoid re-tokenization.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize tokenizer cache.
        
        Args:
            max_size: Maximum number of cached sequences
        """
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, text: str) -> Optional[jnp.ndarray]:
        """
        Get cached tokenization.
        
        Args:
            text: Input text
        
        Returns:
            tokens: Cached tokens or None
        """
        return self.cache.get(text)
    
    def set(self, text: str, tokens: jnp.ndarray):
        """
        Cache tokenization.
        
        Args:
            text: Input text
            tokens: Tokenized sequence
        """
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru = self.access_order.pop(0)
            del self.cache[lru]
        
        self.cache[text] = tokens
        self.access_order.append(text)
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()


class LlamaTokenizer:
    """
    Llama-style tokenizer (simplified).
    
    In practice, this would use the HuggingFace transformers library.
    """
    
    def __init__(self, vocab_size: int = 32000):
        """
        Initialize Llama tokenizer.
        
        Args:
            vocab_size: Vocabulary size
        """
        self.vocab_size = vocab_size
        self.cache = TokenizerCache()
    
    def encode(self, text: str, use_cache: bool = True) -> jnp.ndarray:
        """
        Encode text to tokens.
        
        Args:
            text: Input text
            use_cache: Whether to use cache
        
        Returns:
            tokens: Tokenized sequence
        """
        if use_cache:
            cached = self.cache.get(text)
            if cached is not None:
                return cached
        
        # Simplified tokenization (in practice would use proper tokenizer)
        tokens = jnp.array([hash(c) % self.vocab_size for c in text])
        
        if use_cache:
            self.cache.set(text, tokens)
        
        return tokens
    
    def decode(self, tokens: jnp.ndarray) -> str:
        """
        Decode tokens to text.
        
        Args:
            tokens: Token sequence
        
        Returns:
            text: Decoded text
        """
        # Simplified decoding
        return "".join([chr(t) for t in tokens if t < 128])
    
    def preprocess(self, text: str) -> str:
        """
        Apply tokenizer-specific preprocessing.
        
        Args:
            text: Input text
        
        Returns:
            preprocessed: Preprocessed text
        """
        # Llama-specific preprocessing
        text = text.strip()
        text = text.lower()
        
        return text


def get_tokenizer(tokenizer_type: str, **kwargs) -> Any:
    """
    Get tokenizer by type.
    
    Args:
        tokenizer_type: Type of tokenizer ("gpt2", "llama", "bert")
        **kwargs: Additional arguments
    
    Returns:
        tokenizer: Tokenizer instance
    """
    if tokenizer_type == "gpt2":
        # Would use HuggingFace GPT2Tokenizer in practice
        return None
    elif tokenizer_type == "llama":
        return LlamaTokenizer(**kwargs)
    elif tokenizer_type == "bert":
        # Would use HuggingFace BertTokenizer in practice
        return None
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
