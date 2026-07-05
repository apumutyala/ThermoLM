"""
Sparse Connectivity Patterns for DTM

Defines nested families of sparse interaction graphs for the quadratic EBM.
Sparsity is what makes block Gibbs cheap (few colours, low degree) and what
maps onto locally-connected TSU hardware.

NAMING NOTE: the G8..G24 labels are *loosely inspired by* the graph families
in Extropic's DTM paper (Table I), not a faithful reimplementation of them —
e.g. our 1-D "G8" is a skip-1 chain (2 neighbours/node), not an
8-neighbour graph. Treat the labels as ordered sparsity tiers
(G8 ⊂ G12 ⊂ G16 ⊂ G20 ⊂ G24), nothing more.

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Optional


def generate_connectivity_pattern(
    pattern: str,
    n_vars: int,
    graph_type: str = "banded"
) -> jnp.ndarray:
    """
    Generate a sparse connectivity mask.

    Args:
        pattern: Connectivity pattern name ("G8", "G12", "G16", "G20", "G24")
        n_vars: Number of variables
        graph_type: "grid" (2D neighbourhoods), "banded" (1-D skip-distance
            bands — the sequence-model default), or "chain" (short-range 1-D).
            "bipartite" is accepted as a deprecated alias for "banded": these
            banded graphs are NOT bipartite for G12+ (even skip distances
            connect same-parity nodes), so the old name was misleading.

    Returns:
        connectivity_mask: Boolean adjacency matrix, shape (n_vars, n_vars)
    """
    if graph_type == "grid":
        return _generate_grid_pattern(pattern, n_vars)
    elif graph_type in ("banded", "bipartite"):
        return _generate_banded_pattern(pattern, n_vars)
    elif graph_type == "chain":
        return _generate_chain_pattern(pattern, n_vars)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")


def _generate_grid_pattern(pattern: str, n_vars: int) -> jnp.ndarray:
    """
    Generate 2D grid connectivity patterns for image data.

    Nested 2-D neighbourhoods (labels are sparsity tiers, see module note):
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


def _generate_banded_pattern(pattern: str, n_vars: int) -> jnp.ndarray:
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

    These graphs are NOT bipartite for G12+ (even skips connect same-parity
    nodes) — hence the rename from the old "bipartite" label. An even earlier
    version restricted edges to opposite-parity endpoints so it could 2-colour
    by even/odd index; that filter silently *dropped every even-distance
    skip*, collapsing G12/G16/G20 to G8. The samplers derive a valid colouring
    from the actual graph via
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
        mask: Boolean adjacency matrix (diagonal assumed empty)

    Returns:
        density: Fraction of the n*(n-1) possible directed edges present.
            (An earlier version divided by n^2, counting impossible
            self-loops in the denominator.)
    """
    n_vars = mask.shape[0]
    n_possible = n_vars * (n_vars - 1)
    n_actual = jnp.sum(mask)
    return float(n_actual / n_possible)


def test_connectivity_patterns():
    """Test connectivity pattern generation."""
    print("Testing connectivity patterns...")

    # Test banded patterns
    for pattern in ["G8", "G12", "G16", "G20", "G24"]:
        mask = generate_connectivity_pattern(pattern, n_vars=64, graph_type="banded")
        density = get_connectivity_density(mask)
        print(f"{pattern} banded: density={density:.4f}")

        # Verify symmetric
        assert jnp.array_equal(mask, mask.T), f"{pattern} mask not symmetric"

        # Verify no self-connections
        assert jnp.all(jnp.diagonal(mask) == 0), f"{pattern} has self-connections"

    # Test grid patterns
    for pattern in ["G8", "G12", "G16"]:
        mask = generate_connectivity_pattern(pattern, n_vars=64, graph_type="grid")
        density = get_connectivity_density(mask)
        print(f"{pattern} grid: density={density:.4f}")

    # Deprecated alias still works
    a = generate_connectivity_pattern("G12", 32, graph_type="bipartite")
    b = generate_connectivity_pattern("G12", 32, graph_type="banded")
    assert jnp.array_equal(a, b)

    print("[SUCCESS] Connectivity patterns test passed!")


if __name__ == "__main__":
    test_connectivity_patterns()
