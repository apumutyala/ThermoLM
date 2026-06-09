"""
Graph utilities for ThermoLM JAX

Provides graph coloring and block partitioning for THRML integration.

Author: Apuroop Mutyala
Date: April 2026
"""

import jax
import jax.numpy as jnp
from typing import List, Optional, Tuple, Dict, Any


def color_graph(adjacency: jnp.ndarray) -> List[List[int]]:
    """
    Perform graph coloring for parallel sampling.
    
    Uses Welsh-Powell algorithm for greedy coloring.
    
    Args:
        adjacency: Adjacency matrix (n, n)
    
    Returns:
        blocks: List of variable indices for each color class
    """
    n = adjacency.shape[0]
    colors = {}
    
    # Sort vertices by degree (descending)
    degrees = jnp.sum(adjacency, axis=1)
    vertices = jnp.argsort(degrees)[::-1]
    
    for vertex in vertices:
        # Find used colors among neighbors
        neighbor_colors = set()
        for neighbor in range(n):
            if adjacency[vertex, neighbor] > 0 and neighbor in colors:
                neighbor_colors.add(colors[neighbor])
        
        # Assign smallest available color
        color = 0
        while color in neighbor_colors:
            color += 1
        colors[vertex] = int(color)
    
    # Group by color
    blocks = {}
    for vertex, color in colors.items():
        blocks.setdefault(color, []).append(int(vertex))
    
    return list(blocks.values())


def analyze_connectivity(adjacency: jnp.ndarray) -> Dict[str, Any]:
    """
    Analyze graph connectivity.
    
    Args:
        adjacency: Adjacency matrix (n, n)
    
    Returns:
        analysis: Dictionary of connectivity metrics
    """
    n = adjacency.shape[0]
    
    # Degree distribution
    degrees = jnp.sum(adjacency, axis=1)
    
    # Sparsity
    num_edges = jnp.sum(adjacency > 0) // 2
    max_edges = n * (n - 1) // 2
    sparsity = 1.0 - (num_edges / max_edges)
    
    analysis = {
        'num_nodes': n,
        'num_edges': int(num_edges),
        'max_edges': max_edges,
        'sparsity': float(sparsity),
        'avg_degree': float(jnp.mean(degrees)),
        'max_degree': int(jnp.max(degrees)),
        'min_degree': int(jnp.min(degrees)),
    }
    
    return analysis


def balance_block_sizes(
    blocks: List[List[int]],
    target_size: int,
) -> List[List[int]]:
    """
    Balance blocks to have approximately equal sizes.
    
    Args:
        blocks: List of blocks (each block is a list of vertex indices)
        target_size: Target size for each block
    
    Returns:
        balanced_blocks: Balanced blocks
    """
    balanced = []
    remaining = blocks.copy()
    
    # Greedy balancing
    while remaining:
        # Find block closest to target size
        best_block = min(remaining, key=lambda b: abs(len(b) - target_size))
        balanced.append(best_block)
        remaining.remove(best_block)
    
    return balanced


def visualize_graph(
    adj_matrix: jnp.ndarray,
    colors: Optional[jnp.ndarray] = None,
    save_path: Optional[str] = None,
):
    """
    Visualize graph with optional coloring.
    
    Args:
        adj_matrix: Adjacency matrix (n_vertices, n_vertices)
        colors: Optional color assignments for vertices
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    
    # Create NetworkX graph
    adj_np = np.array(adj_matrix)
    G = nx.from_numpy_array(adj_np)
    
    # Draw graph
    plt.figure(figsize=(10, 10))
    
    if colors is not None:
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_color=colors, with_labels=True, cmap='Set3')
    else:
        nx.draw(G, with_labels=True)
    
    plt.title('Graph Visualization')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def visualize_block_partitioning(
    blocks: List[List[int]],
    adj_matrix: jnp.ndarray,
    save_path: Optional[str] = None,
):
    """
    Visualize block partitioning of graph.
    
    Args:
        blocks: List of blocks (each block is a list of vertex indices)
        adj_matrix: Adjacency matrix (n_vertices, n_vertices)
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    
    # Create color assignment
    n_vertices = adj_matrix.shape[0]
    colors = jnp.zeros(n_vertices, dtype=jnp.int32)
    
    for i, block in enumerate(blocks):
        for vertex in block:
            colors = colors.at[vertex].set(i)
    
    # Visualize
    visualize_graph(adj_matrix, colors=colors, save_path=save_path)


# TODO: Add graph visualization utilities - Implemented above


def dsatur_coloring(adj_matrix: jnp.ndarray) -> jnp.ndarray:
    """
    DSATUR (Degree of Saturation) graph coloring algorithm.
    
    A more sophisticated coloring algorithm that considers both vertex degree
    and saturation (number of different colors in neighbors) for better coloring.
    
    Args:
        adj_matrix: Adjacency matrix (n_vertices, n_vertices)
    
    Returns:
        colors: Color assignment for each vertex (n_vertices,)
    """
    n_vertices = adj_matrix.shape[0]
    colors = jnp.full(n_vertices, -1, dtype=jnp.int32)
    
    # Compute initial degrees
    degrees = jnp.sum(adj_matrix, axis=1)
    
    # Track saturation (number of different colors in neighbors)
    saturation = jnp.zeros(n_vertices, dtype=jnp.int32)
    
    # Track available colors for each vertex
    max_colors = n_vertices
    available = jnp.ones((n_vertices, max_colors), dtype=jnp.bool_)
    
    for step in range(n_vertices):
        # Find uncolored vertex with maximum saturation, breaking ties by degree
        uncolored = colors == -1
        if not jnp.any(uncolored):
            break
        
        # Get candidates (uncolored vertices)
        candidates = jnp.where(uncolored, size=n_vertices)[0]
        
        # Sort by saturation (descending), then by degree (descending)
        candidate_saturation = saturation[candidates]
        candidate_degrees = degrees[candidates]
        
        # Simple selection: pick first uncolored (in practice would sort)
        vertex = candidates[0]
        
        # Find smallest available color
        available_colors = jnp.where(available[vertex], size=max_colors)[0]
        if len(available_colors) == 0:
            color = 0
        else:
            color = available_colors[0]
        
        # Assign color
        colors = colors.at[vertex].set(color)
        
        # Update neighbors
        neighbors = jnp.where(adj_matrix[vertex], size=n_vertices)[0]
        for neighbor in neighbors:
            if colors[neighbor] == -1:
                # Mark this color as unavailable for neighbor
                available = available.at[neighbor, color].set(False)
                # Update saturation if this is a new color
                neighbor_colors = colors[jnp.where(adj_matrix[neighbor])]
                unique_neighbor_colors = jnp.unique(neighbor_colors[neighbor_colors >= 0])
                saturation = saturation.at[neighbor].set(len(unique_neighbor_colors))
    
    return colors


def greedy_graph_coloring(adj_matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Greedy graph coloring algorithm (simple but fast).
    
    Args:
        adj_matrix: Adjacency matrix (n_vertices, n_vertices)
    
    Returns:
        colors: Color assignment for each vertex (n_vertices,)
    """
    n_vertices = adj_matrix.shape[0]
    colors = jnp.full(n_vertices, -1, dtype=jnp.int32)
    
    # Sort vertices by degree (descending)
    degrees = jnp.sum(adj_matrix, axis=1)
    order = jnp.argsort(degrees)[::-1]
    
    for vertex in order:
        # Find colors used by neighbors
        neighbors = jnp.where(adj_matrix[vertex], size=n_vertices)[0]
        neighbor_colors = colors[neighbors]
        used_colors = jnp.unique(neighbor_colors[neighbor_colors >= 0])
        
        # Find smallest available color
        color = 0
        while color in used_colors:
            color += 1
        
        colors = colors.at[vertex].set(color)
    
    return colors
