"""
Denoising Thermodynamic Model (DTM)

Implements the DTM architecture following Extropic.pdf Eq. 7, 8.
DTM chains multiple EBMs to gradually build complexity, overcoming the
mixing-expressivity tradeoff that limits monolithic EBMs.

Design Decision: DTM Chain Architecture
- Rationale: Required by Extropic.pdf to overcome MET
- Impact: Enables complex distributions with simple EBMs
- Trade-off: More complex training (chain of EBMs)
- Downstream: Enables TSU-compatible sampling

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, List
from dataclasses import dataclass

from .quadratic_ebm import QuadraticEBM, QuadraticEBMConfig
from .forward_coupling import ForwardCoupling, ForwardCouplingConfig
from .latent_graph import SparseGraph, LatentGraphConfig


@dataclass
class DTMConfig:
    """Configuration for DTM."""
    n_data_vars: int = 512  # Number of data variables
    n_latent_vars: int = 128  # Number of latent variables
    T: int = 1000  # Number of diffusion steps
    connectivity_pattern: str = "G8"  # Connectivity pattern
    data_connectivity_pattern: str = "G8"  # Data connectivity pattern
    beta: float = 1.0  # Inverse temperature
    gamma_min: float = 0.001  # Minimum coupling strength
    gamma_max: float = 1.0  # Maximum coupling strength
    schedule_type: str = "linear"  # Coupling schedule type
    init_scale: float = 0.01  # Weight initialization scale


class DTM(eqx.Module):
    """
    Denoising Thermodynamic Model following Extropic.pdf.
    
    DTM implements the chain of EBMs:
    P_θ(x_{t-1}|x_t) ∝ e^{-(E_f_{t-1}(x_{t-1},x_t) + E_θ_{t-1}(x_{t-1},z_{t-1},θ))}
    
    where:
    - E_f_{t-1} is the forward process coupling energy
    - E_θ_{t-1} is the learned energy function (quadratic EBM)
    - z_{t-1} are latent variables
    
    The chain of EBMs gradually transforms noise into data, avoiding the
    mixing-expressivity tradeoff that limits monolithic EBMs.
    
    Args:
        ebm_layers: List of quadratic EBM layers (one per timestep or shared)
        forward_coupling: Forward process coupling
        latent_graph: Sparse latent variable graph
        T: Number of diffusion steps
    """
    
    ebm: QuadraticEBM  # Single shared EBM (can be extended to per-timestep EBMs)
    forward_coupling: ForwardCoupling
    latent_graph: SparseGraph
    T: int
    
    def __init__(
        self,
        config: DTMConfig,
        key: jax.random.PRNGKey
    ):
        """Initialize DTM with shared EBM architecture."""
        self.T = config.T
        
        # Initialize forward coupling
        forward_config = ForwardCouplingConfig(
            T=config.T,
            gamma_min=config.gamma_min,
            gamma_max=config.gamma_max,
            schedule_type=config.schedule_type
        )
        self.forward_coupling = ForwardCoupling(forward_config)
        
        # Initialize latent graph
        latent_config = LatentGraphConfig(
            n_data_vars=config.n_data_vars,
            n_latent_vars=config.n_latent_vars,
            connectivity_pattern=config.connectivity_pattern,
            data_connectivity_pattern=config.data_connectivity_pattern
        )
        self.latent_graph = SparseGraph(latent_config)
        
        # Initialize shared quadratic EBM
        # The EBM operates on both data and latent variables
        n_total_vars = config.n_data_vars + config.n_latent_vars
        
        ebm_config = QuadraticEBMConfig(
            n_vars=n_total_vars,
            connectivity_pattern=config.connectivity_pattern,
            beta=config.beta,
            init_scale=config.init_scale
        )
        self.ebm = QuadraticEBM(ebm_config, key)
        
        # Set connectivity mask based on latent graph
        combined_adjacency = self.latent_graph.get_combined_adjacency()
        self.ebm = self.ebm.set_connectivity(combined_adjacency)
    
    def __call__(
        self,
        x_t: jnp.ndarray,
        x_t_minus_1: jnp.ndarray,
        t: int,
        z_t_minus_1: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute energy for DTM transition.
        
        P_θ(x_{t-1}|x_t) ∝ e^{-(E_f_{t-1}(x_{t-1},x_t) + E_θ_{t-1}(x_{t-1},z_{t-1},θ))}
        
        Args:
            x_t: State at timestep t, shape (..., n_total_vars)
            x_t_minus_1: State at timestep t-1, shape (..., n_total_vars)
            t: Timestep index
            z_t_minus_1: Latent variables at timestep t-1 (optional, for future extension)
        
        Returns:
            energy: Total energy for the transition, shape (...)
        """
        # Forward coupling energy
        E_f = self.forward_coupling(x_t, x_t_minus_1, t)
        
        # Learned energy from EBM
        # Currently uses x_t_minus_1 (can be extended to include latents)
        E_theta = self.ebm(x_t_minus_1)
        
        # Total energy
        total_energy = E_f + E_theta
        
        return total_energy
    
    def sample_initial_state(
        self,
        batch_size: int,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """
        Sample initial random state (high temperature / noise).
        
        Args:
            batch_size: Number of samples
            key: Random key
        
        Returns:
            initial_state: Random binary spins, shape (batch_size, n_total_vars)
        """
        n_total_vars = self.latent_graph.get_n_total_vars()
        
        # Sample random binary spins {-1, 1}
        initial_state = jax.random.randint(
            key,
            (batch_size, n_total_vars),
            minval=0,
            maxval=2
        ) * 2 - 1
        
        return initial_state
    
    def get_energy_components(
        self,
        x_t: jnp.ndarray,
        x_t_minus_1: jnp.ndarray,
        t: int
    ) -> tuple:
        """
        Get separate energy components for analysis.
        
        Args:
            x_t: State at timestep t
            x_t_minus_1: State at timestep t-1
            t: Timestep index
        
        Returns:
            E_f: Forward coupling energy
            E_theta: Learned energy
            E_theta_pairwise: Pairwise component of learned energy
            E_theta_unary: Unary component of learned energy
        """
        E_f = self.forward_coupling(x_t, x_t_minus_1, t)
        E_theta = self.ebm(x_t_minus_1)
        E_theta_pairwise, E_theta_unary = self.ebm.get_energy_components(x_t_minus_1)
        
        return E_f, E_theta, E_theta_pairwise, E_theta_unary


def test_dtm():
    """Test DTM implementation."""
    print("Testing DTM...")
    
    config = DTMConfig(
        n_data_vars=64,
        n_latent_vars=16,
        T=100
    )
    
    key = jax.random.PRNGKey(0)
    dtm = DTM(config, key)
    
    # Test initial state sampling
    initial_state = dtm.sample_initial_state(batch_size=4, key=key)
    expected_shape = (4, 80)  # 64 data + 16 latent
    assert initial_state.shape == expected_shape, f"Expected {expected_shape}, got {initial_state.shape}"
    print(f"Initial state shape: {initial_state.shape}")
    print(f"Initial state values: {initial_state[0, :5]}...")
    
    # Test energy computation
    x_t = dtm.sample_initial_state(batch_size=4, key=jax.random.PRNGKey(1))
    x_t_minus_1 = dtm.sample_initial_state(batch_size=4, key=jax.random.PRNGKey(2))
    
    energy = dtm(x_t, x_t_minus_1, t=50)
    assert energy.shape == (4,), f"Expected shape (4,), got {energy.shape}"
    print(f"Energy shape: {energy.shape}")
    print(f"Energy values: {energy}")
    
    # Test energy components
    E_f, E_theta, E_theta_pairwise, E_theta_unary = dtm.get_energy_components(x_t, x_t_minus_1, t=50)
    print(f"Forward coupling energy: {E_f}")
    print(f"Learned energy: {E_theta}")
    print(f"Pairwise energy: {E_theta_pairwise}")
    print(f"Unary energy: {E_theta_unary}")
    
    # Verify total energy
    total_energy_computed = E_f + E_theta
    assert jnp.allclose(energy, total_energy_computed), "Energy components don't sum to total"
    
    print("[SUCCESS] DTM test passed!")


if __name__ == "__main__":
    test_dtm()
