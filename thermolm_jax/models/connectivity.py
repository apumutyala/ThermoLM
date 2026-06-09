"""
Sparse Connectivity Patterns for DTM

Implements sparse connectivity patterns from Extropic.pdf Table I.
These patterns define which variables interact in the quadratic EBM,
enabling efficient Gibbs sampling and TSU hardware compatibility.

Design Decision: Sparse Connectivity Patterns
- Rationale: Required by Extropic.pdf for TSU hardware efficiency
- Impact: Limits interactions to local neighbors
- Trade-off: Limited expressivity vs energy efficiency
- Downstream: Enables parallel block Gibbs sampling

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Optional


def generate_connectivity_pattern(
    pattern: str,
    n_vars: int,
    graph_type: str = "bipartite"
) -> jnp.ndarray:
    """
    Generate sparse connectivity patterns from Extropic.pdf Table I.
    
    Args:
        pattern: Connectivity pattern name ("G8", "G12", "G16", "G20", "G24")
        n_vars: Number of variables
        graph_type: Type of graph ("grid" for 2D images, "bipartite" for language)
    
    Returns:
        connectivity_mask: Boolean adjacency matrix, shape (n_vars, n_vars)
    """
    if graph_type == "grid":
        return _generate_grid_pattern(pattern, n_vars)
    elif graph_type == "bipartite":
        return _generate_bipartite_pattern(pattern, n_vars)
    elif graph_type == "chain":
        return _generate_chain_pattern(pattern, n_vars)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")


def _generate_grid_pattern(pattern: str, n_vars: int) -> jnp.ndarray:
    """
    Generate 2D grid connectivity patterns for image data.
    
    Patterns from Extropic.pdf Table I:
    - G8: Nearest neighbors on 2D grid (4-connectivity)
    - G12: G8 + diagonal neighbors (8-connectivity)
    - G16: G12 + next-nearest neighbors
    - G20: G16 + additional diagonals
    - G24: Full 5x5 neighborhood
    """
    # Assume square grid
    grid_size = int(jnp.sqrt(n_vars))
    if grid_size * grid_size != n_vars:
        raise ValueError(f"n_vars={n_vars} must be perfect square for grid pattern")
    
    mask = jnp.zeros((n_vars, n_vars), dtype=bool)
    
    # Convert linear index to (row, col)
    def to_2d(idx):
        return idx // grid_size, idx % grid_size
    
    # Connectivity definitions
    patterns = {
        "G8": [(0, 1), (1, 0)],  # Right, Down
        "G12": [(0, 1), (1, 0), (1, 1), (1, -1)],  # + diagonals
        "G16": [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0)],  # + next-nearest
        "G20": [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0), (2, 1), (2, -1)],
        "G24": [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0), (2, 1), (2, -1), (2, 2), (2, -2), (0, -2), (-2, 0)]
    }
    
    if pattern not in patterns:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    offsets = patterns[pattern]
    
    for i in range(n_vars):
        row_i, col_i = to_2d(i)
        for dr, dc in offsets:
            row_j = row_i + dr
            col_j = col_i + dc
            
            if 0 <= row_j < grid_size and 0 <= col_j < grid_size:
                j = row_j * grid_size + col_j
                mask = mask.at[i, j].set(True)
                mask = mask.at[j, i].set(True)
    
    return mask


def _generate_bipartite_pattern(pattern: str, n_vars: int) -> jnp.ndarray:
    """
    Generate banded (skip-distance) 1D connectivity patterns.

    Each node connects to neighbours at a set of skip distances, giving a nested
    family G8 ⊂ G12 ⊂ G16 ⊂ G20 ⊂ G24 of increasing range.

    Patterns:
    - G8:  skip 1            (adjacent)
    - G12: skips 1, 2
    - G16: skips 1, 2, 4
    - G20: skips 1, 2, 4, 8
    - G24: all distances     (dense)

    NOTE: an earlier version restricted edges to opposite-parity endpoints so it
    could 2-colour by even/odd index. That filter silently *dropped every
    even-distance skip*, collapsing G12/G16/G20 to G8. The restriction is
    removed here; the samplers no longer assume an even/odd colouring — they
    derive a valid colouring from the graph via
    ``thermolm_jax.sampling.chromatic_gibbs.greedy_coloring``.
    """
    mask = jnp.zeros((n_vars, n_vars), dtype=bool)

    pattern_skips = {
        "G8": [1],
        "G12": [1, 2],
        "G16": [1, 2, 4],
        "G20": [1, 2, 4, 8],
        "G24": list(range(1, n_vars)),  # all distances
    }

    if pattern not in pattern_skips:
        raise ValueError(f"Unknown pattern: {pattern}")

    skips = pattern_skips[pattern]

    for i in range(n_vars):
        for skip in skips:
            for j in [i + skip, i - skip]:
                if 0 <= j < n_vars:
                    mask = mask.at[i, j].set(True)
                    mask = mask.at[j, i].set(True)

    return mask


def _generate_chain_pattern(pattern: str, n_vars: int) -> jnp.ndarray:
    """
    Generate chain connectivity patterns for sequential data.
    
    Simple 1D chain with local connections.
    """
    mask = jnp.zeros((n_vars, n_vars), dtype=bool)
    
    pattern_neighbors = {
        "G8": [1],
        "G12": [1, 2],
        "G16": [1, 2, 3],
        "G20": [1, 2, 3, 4],
        "G24": list(range(1, min(n_vars, 10)))
    }
    
    if pattern not in pattern_neighbors:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    neighbors = pattern_neighbors[pattern]
    
    for i in range(n_vars):
        for neighbor in neighbors:
            for j in [i + neighbor, i - neighbor]:
                if 0 <= j < n_vars:
                    mask = mask.at[i, j].set(True)
                    mask = mask.at[j, i].set(True)
    
    return mask


def get_connectivity_density(mask: jnp.ndarray) -> float:
    """
    Compute the density of connectivity (fraction of possible edges present).
    
    Args:
        mask: Boolean adjacency matrix
    
    Returns:
        density: Fraction of non-zero edges
    """
    n_vars = mask.shape[0]
    n_possible = n_vars * n_vars
    n_actual = jnp.sum(mask)
    return float(n_actual / n_possible)


def get_bipartite_coloring(n_vars: int) -> Tuple[List[int], List[int]]:
    """
    Get bipartite coloring for block Gibbs sampling.
    
    For a bipartite graph, returns two color blocks (even and odd indices).
    This enables parallel updates of all even positions, then all odd positions.
    
    Args:
        n_vars: Number of variables
    
    Returns:
        color0: Indices for color 0 (even positions)
        color1: Indices for color 1 (odd positions)
    """
    color0 = [i for i in range(n_vars) if i % 2 == 0]
    color1 = [i for i in range(n_vars) if i % 2 == 1]
    return color0, color1


def test_connectivity_patterns():
    """Test connectivity pattern generation."""
    print("Testing connectivity patterns...")
    
    # Test bipartite patterns
    for pattern in ["G8", "G12", "G16", "G20", "G24"]:
        mask = generate_connectivity_pattern(pattern, n_vars=64, graph_type="bipartite")
        density = get_connectivity_density(mask)
        print(f"{pattern} bipartite: density={density:.4f}")
        
        # Verify symmetric
        assert jnp.array_equal(mask, mask.T), f"{pattern} mask not symmetric"
        
        # Verify no self-connections
        assert jnp.all(jnp.diagonal(mask) == 0), f"{pattern} has self-connections"
    
    # Test grid patterns
    for pattern in ["G8", "G12", "G16"]:
        mask = generate_connectivity_pattern(pattern, n_vars=64, graph_type="grid")
        density = get_connectivity_density(mask)
        print(f"{pattern} grid: density={density:.4f}")
    
    # Test bipartite coloring
    color0, color1 = get_bipartite_coloring(64)
    print(f"Bipartite coloring: color0={len(color0)} nodes, color1={len(color1)} nodes")
    assert len(color0) + len(color1) == 64
    
    print("[SUCCESS] Connectivity patterns test passed!")


if __name__ == "__main__":
    test_connectivity_patterns()
