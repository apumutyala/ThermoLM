"""
Metrics Module for ThermoLM JAX

Provides evaluation metrics for language models.

Author: Apuroop Mutyala
Date: April 2026
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, List


def compute_perplexity(
    model: Any,
    dataloader: jnp.ndarray,
) -> float:
    """
    Compute perplexity on validation set.
    
    PPL = exp(-1/N Σ log p(x_i))
    
    Args:
        model: Trained model
        dataloader: Validation data (batched)
    
    Returns:
        perplexity: Perplexity score
    """
    total_log_prob = 0.0
    total_tokens = 0
    
    for batch in dataloader:
        # TODO: Compute log probability
        log_prob = model.compute_log_prob(batch)
        total_log_prob += jnp.sum(log_prob)
        total_tokens += batch.size
    
    perplexity = jnp.exp(-total_log_prob / total_tokens)
    return float(perplexity)


def compute_generation_quality(
    model: Any,
    prompts: List[str],
    references: List[str],
    num_samples: int = 100,
) -> Dict[str, float]:
    """
    Evaluate generation quality.
    
    Metrics:
    - BLEU score
    - Diversity metrics
    - Coherence score
    
    Args:
        model: Trained model
        prompts: List of prompt strings
        references: List of reference strings
        num_samples: Number of samples per prompt
    
    Returns:
        metrics: Dictionary of metrics
    """
    generations = []
    for prompt in prompts:
        for _ in range(num_samples):
            gen = model.generate(prompt)
            generations.append(gen)
    
    bleu = compute_bleu(generations, references)
    diversity = compute_diversity(generations)
    coherence = compute_coherence(generations)
    
    return {
        'bleu': bleu,
        'diversity': diversity,
        'coherence': coherence,
    }


def compute_bleu(
    predictions: jnp.ndarray,
    references: jnp.ndarray,
    max_order: int = 4,
) -> jnp.ndarray:
    """
    Compute BLEU score.
    
    Args:
        predictions: Generated sequences (batch, seq_len)
        references: Reference sequences (batch, seq_len)
        max_order: Maximum n-gram order
    
    Returns:
        bleu: BLEU score
    """
    from collections import Counter
    import math
    
    predictions = jnp.array(predictions)
    references = jnp.array(references)
    
    bleu_scores = []
    
    for pred, ref in zip(predictions, references):
        # Remove padding
        pred = pred[pred != 0]
        ref = ref[ref != 0]
        
        # Compute n-gram precision
        precisions = []
        for n in range(1, max_order + 1):
            pred_ngrams = []
            ref_ngrams = []
            
            for i in range(len(pred) - n + 1):
                pred_ngrams.append(tuple(pred[i:i+n]))
            
            for i in range(len(ref) - n + 1):
                ref_ngrams.append(tuple(ref[i:i+n]))
            
            if len(pred_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            pred_counter = Counter(pred_ngrams)
            ref_counter = Counter(ref_ngrams)
            
            # Count matches
            matches = 0
            for ngram, count in pred_counter.items():
                matches += min(count, ref_counter.get(ngram, 0))
            
            precision = matches / len(pred_ngrams)
            precisions.append(precision)
        
        # Compute brevity penalty
        pred_len = len(pred)
        ref_len = len(ref)
        
        if pred_len > ref_len:
            bp = 1.0
        elif pred_len == 0:
            bp = 0.0
        else:
            bp = math.exp(1 - ref_len / pred_len)
        
        # Compute geometric mean of precisions
        if all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            geo_mean = 0.0
        
        bleu = bp * geo_mean
        bleu_scores.append(bleu)
    
    return jnp.array(bleu_scores).mean()


def compute_diversity(
    predictions: jnp.ndarray,
    n: int = 2,
) -> jnp.ndarray:
    """
    Compute diversity (unique n-grams).
    
    Args:
        predictions: Generated sequences (batch, seq_len)
        n: N-gram size
    
    Returns:
        diversity: Diversity score
    """
    from collections import Counter
    
    predictions = np.array(predictions)
    
    all_ngrams = []
    
    for pred in predictions:
        pred = pred[pred != 0]
        
        for i in range(len(pred) - n + 1):
            ngram = tuple(pred[i:i+n])
            all_ngrams.append(ngram)
    
    if len(all_ngrams) == 0:
        return jnp.array(0.0)
    
    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)
    
    diversity = unique_ngrams / total_ngrams
    
    return jnp.array(diversity)


def compute_coherence(
    predictions: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute coherence (self-BLEU across sequence).
    
    Args:
        predictions: Generated sequences (batch, seq_len)
    
    Returns:
        coherence: Coherence score
    """
    from collections import Counter
    import math
    
    predictions = np.array(predictions)
    
    coherence_scores = []
    
    for pred in predictions:
        pred = pred[pred != 0]
        
        if len(pred) < 4:
            coherence_scores.append(0.0)
            continue
        
        # Split into halves
        mid = len(pred) // 2
        first_half = pred[:mid]
        second_half = pred[mid:]
        
        # Compute BLEU between halves
        precisions = []
        for n in [1, 2]:
            first_ngrams = []
            second_ngrams = []
            
            for i in range(len(first_half) - n + 1):
                first_ngrams.append(tuple(first_half[i:i+n]))
            
            for i in range(len(second_half) - n + 1):
                second_ngrams.append(tuple(second_half[i:i+n]))
            
            if len(first_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            first_counter = Counter(first_ngrams)
            second_counter = Counter(second_ngrams)
            
            matches = 0
            for ngram, count in first_counter.items():
                matches += min(count, second_counter.get(ngram, 0))
            
            precision = matches / len(first_ngrams)
            precisions.append(precision)
        
        if all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            geo_mean = 0.0
        
        coherence_scores.append(geo_mean)
    
    return jnp.array(coherence_scores).mean()


def compute_bleu_text(generations: List[str], references: List[str]) -> float:
    """
    Compute BLEU score for text strings.

    Args:
        generations: Generated texts
        references: Reference texts

    Returns:
        bleu: BLEU score
    """
    from collections import Counter
    import math

    # Tokenize texts
    def tokenize(text):
        return text.lower().split()

    # Compute BLEU for each generation-reference pair
    bleu_scores = []
    for gen, ref in zip(generations, references):
        gen_tokens = tokenize(gen)
        ref_tokens = tokenize(ref)

        # Compute n-gram precision
        precisions = []
        for n in range(1, 5):  # 1-gram to 4-gram
            gen_ngrams = []
            for i in range(len(gen_tokens) - n + 1):
                gen_ngrams.append(tuple(gen_tokens[i:i+n]))

            ref_ngrams = []
            for i in range(len(ref_tokens) - n + 1):
                ref_ngrams.append(tuple(ref_tokens[i:i+n]))

            if len(gen_ngrams) == 0:
                precisions.append(0.0)
                continue

            gen_counter = Counter(gen_ngrams)
            ref_counter = Counter(ref_ngrams)

            # Count matches
            matches = 0
            for ngram, count in gen_counter.items():
                matches += min(count, ref_counter.get(ngram, 0))

            precision = matches / len(gen_ngrams)
            precisions.append(precision)

        # Compute brevity penalty
        gen_len = len(gen_tokens)
        ref_len = len(ref_tokens)

        if gen_len > ref_len:
            bp = 1.0
        elif gen_len == 0:
            bp = 0.0
        else:
            bp = math.exp(1 - ref_len / gen_len)

        # Compute geometric mean of precisions
        if all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            geo_mean = 0.0

        bleu = bp * geo_mean
        bleu_scores.append(bleu)

    return sum(bleu_scores) / len(bleu_scores)


def compute_diversity(generations: List[str]) -> float:
    """
    Compute diversity metrics for text strings.

    Args:
        generations: Generated texts

    Returns:
        diversity: Diversity score (ratio of unique n-grams to total n-grams)
    """
    from collections import Counter

    def tokenize(text):
        return text.lower().split()

    # Collect all n-grams across all generations
    all_ngrams = []
    n = 2  # Use bigrams for diversity

    for gen in generations:
        tokens = tokenize(gen)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            all_ngrams.append(ngram)

    if len(all_ngrams) == 0:
        return 0.0

    # Compute diversity as ratio of unique n-grams to total n-grams
    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)

    diversity = unique_ngrams / total_ngrams

    return diversity


def compute_coherence(generations: List[str]) -> float:
    """
    Compute coherence score for text strings using self-BLEU.

    Args:
        generations: Generated texts

    Returns:
        coherence: Coherence score (average self-BLEU across sentences)
    """
    from collections import Counter
    import math

    def tokenize(text):
        return text.lower().split()

    # Split each generation into sentences (simple split by period)
    def split_sentences(text):
        return [s.strip() for s in text.split('.') if s.strip()]

    coherence_scores = []

    for gen in generations:
        sentences = split_sentences(gen)
        if len(sentences) < 2:
            coherence_scores.append(1.0)  # Single sentence is coherent
            continue

        # Compute self-BLEU for each sentence against the rest
        sentence_coherence = []
        for i, sent in enumerate(sentences):
            # Use the rest of the text as reference
            ref_text = ' '.join(sentences[:i] + sentences[i+1:])
            if not ref_text:
                sentence_coherence.append(1.0)
                continue

            # Compute BLEU between sentence and reference
            sent_tokens = tokenize(sent)
            ref_tokens = tokenize(ref_text)

            # Compute n-gram precision
            precisions = []
            for n in range(1, 3):  # Use bigrams for coherence
                sent_ngrams = []
                for j in range(len(sent_tokens) - n + 1):
                    sent_ngrams.append(tuple(sent_tokens[j:j+n]))

                ref_ngrams = []
                for j in range(len(ref_tokens) - n + 1):
                    ref_ngrams.append(tuple(ref_tokens[j:j+n]))

                if len(sent_ngrams) == 0:
                    precisions.append(0.0)
                    continue

                sent_counter = Counter(sent_ngrams)
                ref_counter = Counter(ref_ngrams)

                matches = 0
                for ngram, count in sent_counter.items():
                    matches += min(count, ref_counter.get(ngram, 0))

                precision = matches / len(sent_ngrams)
                precisions.append(precision)

            if all(p > 0 for p in precisions):
                geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
            else:
                geo_mean = 0.0

            sentence_coherence.append(geo_mean)

        if sentence_coherence:
            coherence_scores.append(sum(sentence_coherence) / len(sentence_coherence))
        else:
            coherence_scores.append(1.0)

    return sum(coherence_scores) / len(coherence_scores)


# TODO: Implement all metrics
# TODO: Add more metrics (ROUGE, METEOR, etc.)
