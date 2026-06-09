"""
Sparse Latent Variable Graph for DTM

Implements sparse latent variable graph structure for DTM.
Latent variables increase model complexity independently of data dimension,
following Extropic.pdf's approach for overcoming the mixing-expressivity tradeoff.

Design Decision: Fixed Latent Graph with Sparse Connectivity
- Rationale: Balance simplicity and expressivity
- Impact: Allows independent scaling of model complexity
- Trade-off: Fixed structure vs learned structure
- Downstream: Can experiment with learned graphs later

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class LatentGraphConfig:
    """Configuration for latent variable graph."""
    n_data_vars: int = 512  # Number of data variables
    n_latent_vars: int = 128  # Number of latent variables
    latent_ratio: float = 0.25  # Latent to data variable ratio
    connectivity_pattern: str = "G8"  # Connectivity pattern for latent graph
    data_connectivity_pattern: str = "G8"  # Connectivity pattern for data graph


class SparseGraph(eqx.Module):
    """
    Sparse latent variable graph for DTM.
    
    The graph consists of:
    - Data nodes: Binary spins from language embedding
    - Latent nodes: Auxiliary binary spins for expressivity
    - Data-data edges: Sparse local connections (for language structure)
    - Data-latent edges: Sparse connections (for expressivity)
    - Latent-latent edges: Sparse connections (for latent structure)
    
    Args:
        is_data: Boolean mask indicating which nodes are data vs latent
        data_adjacency: Adjacency matrix for data-data connections
        latent_adjacency: Adjacency matrix for latent-latent connections
        data_latent_adjacency: Adjacency matrix for data-latent connections
    """
    
    is_data: jnp.ndarray  # Shape (n_total_vars,)
    data_adjacency: jnp.ndarray  # Shape (n_total_vars, n_total_vars)
    latent_adjacency: jnp.ndarray  # Shape (n_total_vars, n_total_vars)
    data_latent_adjacency: jnp.ndarray  # Shape (n_total_vars, n_total_vars)
    
    def __init__(
        self,
        config: LatentGraphConfig,
        data_connectivity_mask: Optional[jnp.ndarray] = None,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0)
    ):
        """Initialize sparse graph structure."""
        n_data = config.n_data_vars
        n_latent = config.n_latent_vars
        n_total = n_data + n_latent
        
        # Mark which nodes are data vs latent
        self.is_data = jnp.concatenate([
            jnp.ones(n_data, dtype=bool),
            jnp.zeros(n_latent, dtype=bool)
        ])
        
        # Initialize adjacency matrices
        self.data_adjacency = jnp.zeros((n_total, n_total), dtype=bool)
        self.latent_adjacency = jnp.zeros((n_total, n_total), dtype=bool)
        self.data_latent_adjacency = jnp.zeros((n_total, n_total), dtype=bool)
        
        # Set up data-data connections (use provided mask or generate)
        if data_connectivity_mask is not None:
            # Only set data-data portion
            self.data_adjacency = self.data_adjacency.at[:n_data, :n_data].set(
                data_connectivity_mask
            )
        else:
            # Generate bipartite pattern for language
            from .connectivity import generate_connectivity_pattern
            data_mask = generate_connectivity_pattern(
                config.data_connectivity_pattern,
                n_data,
                graph_type="bipartite"
            )
            self.data_adjacency = self.data_adjacency.at[:n_data, :n_data].set(data_mask)
        
        # Set up latent-latent connections (sparse)
        from .connectivity import generate_connectivity_pattern
        latent_mask = generate_connectivity_pattern(
            config.connectivity_pattern,
            n_latent,
            graph_type="bipartite"
        )
        self.latent_adjacency = self.latent_adjacency.at[n_data:, n_data:].set(latent_mask)
        
        # Set up data-latent connections (sparse random)
        key_data_latent = jax.random.split(key)[0]
        # Connect each data node to a few random latent nodes
        n_connections_per_data = max(1, n_latent // 10)  # 10% of latent nodes
        for i in range(n_data):
            latent_indices = jax.random.choice(
                key_data_latent,
                n_latent,
                (n_connections_per_data,),
                replace=False
            )
            for j in latent_indices:
                j_abs = int(n_data + j)
                self.data_latent_adjacency = self.data_latent_adjacency.at[i, j_abs].set(True)
                self.data_latent_adjacency = self.data_latent_adjacency.at[j_abs, i].set(True)
    
    def get_data_nodes(self) -> jnp.ndarray:
        """Get indices of data nodes."""
        return jnp.where(self.is_data)[0]
    
    def get_latent_nodes(self) -> jnp.ndarray:
        """Get indices of latent nodes."""
        return jnp.where(~self.is_data)[0]
    
    def get_combined_adjacency(self) -> jnp.ndarray:
        """
        Get combined adjacency matrix including all connection types.
        
        Returns:
            combined_adjacency: Combined sparse adjacency matrix
        """
        return (self.data_adjacency | self.latent_adjacency | self.data_latent_adjacency)
    
    def get_n_data_vars(self) -> int:
        """Get number of data variables."""
        return int(jnp.sum(self.is_data))
    
    def get_n_latent_vars(self) -> int:
        """Get number of latent variables."""
        return int(jnp.sum(~self.is_data))
    
    def get_n_total_vars(self) -> int:
        """Get total number of variables."""
        return len(self.is_data)


def test_sparse_graph():
    """Test sparse graph implementation."""
    print("Testing SparseGraph...")
    
    config = LatentGraphConfig(
        n_data_vars=64,
        n_latent_vars=16,
        connectivity_pattern="G8"
    )
    
    graph = SparseGraph(config)
    
    # Test node counts
    n_data = graph.get_n_data_vars()
    n_latent = graph.get_n_latent_vars()
    n_total = graph.get_n_total_vars()
    
    assert n_data == 64, f"Expected 64 data vars, got {n_data}"
    assert n_latent == 16, f"Expected 16 latent vars, got {n_latent}"
    assert n_total == 80, f"Expected 80 total vars, got {n_total}"
    
    print(f"Data vars: {n_data}")
    print(f"Latent vars: {n_latent}")
    print(f"Total vars: {n_total}")
    
    # Test node indices
    data_indices = graph.get_data_nodes()
    latent_indices = graph.get_latent_nodes()
    
    assert len(data_indices) == 64, f"Expected 64 data indices, got {len(data_indices)}"
    assert len(latent_indices) == 16, f"Expected 16 latent indices, got {len(latent_indices)}"
    
    print(f"Data indices: {data_indices[:5]}...")
    print(f"Latent indices: {latent_indices}")
    
    # Test adjacency matrices
    combined_adj = graph.get_combined_adjacency()
    n_edges = jnp.sum(combined_adj)
    density = n_edges / (n_total * n_total)
    
    print(f"Number of edges: {n_edges}")
    print(f"Graph density: {density:.4f}")
    
    # Verify no self-connections
    assert jnp.all(jnp.diagonal(combined_adj) == 0), "Graph has self-connections"
    
    # Verify symmetry
    assert jnp.array_equal(combined_adj, combined_adj.T), "Graph is not symmetric"
    
    print("[SUCCESS] SparseGraph test passed!")


if __name__ == "__main__":
    test_sparse_graph()
