"""
Comparison Module for ThermoLM JAX

Provides utilities for comparing model performance.

Author: Apuroop Mutyala
Date: April 2026
"""

from typing import Dict, Any, List


def compare_metrics(
    metrics1: Dict[str, float],
    metrics2: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """
    Compare two sets of metrics.
    
    Args:
        metrics1: First set of metrics
        metrics2: Second set of metrics
    
    Returns:
        comparison: Nested dictionary with comparisons
    """
    comparison = {
        'model1': metrics1,
        'model2': metrics2,
        'ratios': {},
        'differences': {},
    }
    
    for key in metrics1:
        if key in metrics2:
            comparison['ratios'][key] = metrics1[key] / metrics2[key]
            comparison['differences'][key] = metrics1[key] - metrics2[key]
    
    return comparison


def compare_multiple(
    metrics_list: List[Dict[str, float]],
    names: List[str],
) -> Dict[str, Any]:
    """
    Compare multiple sets of metrics.
    
    Args:
        metrics_list: List of metric dictionaries
        names: Names of models
    
    Returns:
        comparison: Comparison results
    """
    comparison = {
        'names': names,
        'metrics': metrics_list,
        'best': {},
        'worst': {},
    }
    
    # Find best and worst for each metric
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    for key in all_keys:
        values = [m.get(key, float('inf')) for m in metrics_list]
        comparison['best'][key] = min(values)
        comparison['worst'][key] = max(values)
    
    return comparison


# TODO: Add statistical significance testing - Implemented below
# TODO: Implement visualization of comparisons - Implemented below


def statistical_significance_test(
    metrics_a: jnp.ndarray,
    metrics_b: jnp.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform statistical significance test between two sets of metrics.
    
    Args:
        metrics_a: Metrics from model A (n_samples,)
        metrics_b: Metrics from model B (n_samples,)
        alpha: Significance level
    
    Returns:
        results: Dictionary with test results
    """
    from scipy import stats
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(metrics_a, metrics_b)
    
    # Compute effect size (Cohen's d)
    mean_diff = jnp.mean(metrics_a - metrics_b)
    pooled_std = jnp.sqrt((jnp.var(metrics_a) + jnp.var(metrics_b)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
    
    # Determine significance
    significant = p_value < alpha
    
    results = {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': significant,
        'alpha': alpha,
        'effect_size': float(cohens_d),
        'mean_a': float(jnp.mean(metrics_a)),
        'mean_b': float(jnp.mean(metrics_b)),
        'std_a': float(jnp.std(metrics_a)),
        'std_b': float(jnp.std(metrics_b)),
    }
    
    return results


def visualize_comparison(
    metrics_a: jnp.ndarray,
    metrics_b: jnp.ndarray,
    model_name_a: str,
    model_name_b: str,
    metric_name: str,
    save_path: Optional[str] = None,
):
    """
    Visualize comparison between two models.
    
    Args:
        metrics_a: Metrics from model A (n_samples,)
        metrics_b: Metrics from model B (n_samples,)
        model_name_a: Name of model A
        model_name_b: Name of model B
        metric_name: Name of metric being compared
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    metrics_a = np.array(metrics_a)
    metrics_b = np.array(metrics_b)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Box plot
    axes[0].boxplot([metrics_a, metrics_b], labels=[model_name_a, model_name_b])
    axes[0].set_ylabel(metric_name)
    axes[0].set_title('Distribution Comparison')
    
    # Histogram
    axes[1].hist(metrics_a, alpha=0.5, label=model_name_a, bins=30)
    axes[1].hist(metrics_b, alpha=0.5, label=model_name_b, bins=30)
    axes[1].set_xlabel(metric_name)
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Histogram Comparison')
    axes[1].legend()
    
    # Scatter plot (paired)
    axes[2].scatter(metrics_a, metrics_b, alpha=0.5)
    axes[2].plot([metrics_a.min(), metrics_a.max()], [metrics_a.min(), metrics_a.max()], 'r--', label='y=x')
    axes[2].set_xlabel(f'{model_name_a} {metric_name}')
    axes[2].set_ylabel(f'{model_name_b} {metric_name}')
    axes[2].set_title('Paired Comparison')
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()
