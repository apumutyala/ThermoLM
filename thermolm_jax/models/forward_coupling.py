"""
Forward Process Coupling for DTM

Implements the forward process energy E_f_{t-1} from Extropic.pdf Eq. C1.
This couples consecutive timesteps in the diffusion process.

Design Decision: Linear Forward Coupling Schedule
- Rationale: Follows Extropic.pdf for simplicity
- Impact: Simple element-wise coupling between timesteps
- Trade-off: Linear schedule may not be optimal for all data
- Downstream: Can experiment with different schedules later

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class ForwardCouplingConfig:
    """Configuration for forward process coupling."""
    T: int = 1000  # Number of diffusion steps
    gamma_min: float = 0.001  # Minimum coupling strength
    gamma_max: float = 1.0  # Maximum coupling strength
    schedule_type: str = "linear"  # Schedule type (linear, cosine)


class ForwardCoupling(eqx.Module):
    """
    Forward process energy E_f_{t-1} from Extropic.pdf Eq. C1.
    
    Coupling energy: E_f_{t-1} = Σ_i Γ_i(t)/2 * x_t_i * x_{t-1}_i
    
    where:
    - Γ_i(t) is the coupling strength schedule
    - x_t_i is the state at timestep t
    - x_{t-1}_i is the state at timestep t-1
    
    This couples consecutive timesteps, enabling the diffusion process
    to gradually transform noise into data.
    
    Args:
        gamma: Coupling strength schedule, shape (T,)
    """
    
    gamma: jnp.ndarray  # Coupling strength schedule
    
    def __init__(self, config: ForwardCouplingConfig):
        """Initialize forward coupling with schedule."""
        if config.schedule_type == "linear":
            self.gamma = self._linear_schedule(
                config.T, 
                config.gamma_min, 
                config.gamma_max
            )
        elif config.schedule_type == "cosine":
            self.gamma = self._cosine_schedule(
                config.T,
                config.gamma_min,
                config.gamma_max
            )
        else:
            raise ValueError(f"Unknown schedule type: {config.schedule_type}")
    
    def _linear_schedule(
        self, 
        T: int, 
        gamma_min: float, 
        gamma_max: float
    ) -> jnp.ndarray:
        """
        Linear schedule for coupling strength.
        
        Γ(t) = gamma_min + (gamma_max - gamma_min) * t / (T - 1)
        
        Args:
            T: Number of steps
            gamma_min: Minimum coupling strength
            gamma_max: Maximum coupling strength
        
        Returns:
            gamma: Coupling strength schedule, shape (T,)
        """
        t = jnp.arange(T)
        gamma = gamma_min + (gamma_max - gamma_min) * t / (T - 1)
        return gamma
    
    def _cosine_schedule(
        self,
        T: int,
        gamma_min: float,
        gamma_max: float
    ) -> jnp.ndarray:
        """
        Cosine schedule for coupling strength.
        
        Γ(t) = gamma_min + 0.5 * (gamma_max - gamma_min) * (1 + cos(π * t / (T - 1)))
        
        Args:
            T: Number of steps
            gamma_min: Minimum coupling strength
            gamma_max: Maximum coupling strength
        
        Returns:
            gamma: Coupling strength schedule, shape (T,)
        """
        t = jnp.arange(T)
        gamma = gamma_min + 0.5 * (gamma_max - gamma_min) * (1 + jnp.cos(jnp.pi * t / (T - 1)))
        return gamma
    
    def __call__(
        self,
        x_t: jnp.ndarray,
        x_t_minus_1: jnp.ndarray,
        t: int
    ) -> jnp.ndarray:
        """
        Compute forward coupling energy.
        
        Args:
            x_t: State at timestep t, shape (..., n_vars)
            x_t_minus_1: State at timestep t-1, shape (..., n_vars)
            t: Timestep index
        
        Returns:
            energy: Coupling energy, shape (...)
        """
        gamma_t = self.gamma[t]

        # Ferromagnetic coupling: aligned consecutive states have LOWER energy,
        # so the chain prefers x_t ≈ x_{t-1} (a denoising/smoothing prior).
        #   E_f = -(Γ_t / 2) Σ_i x_t,i x_{t-1},i
        # The earlier version returned +Σ (no minus), which rewarded *anti*-
        # aligned states — the opposite of a denoising coupling, and the
        # opposite sign to the THRML factor branch (to_thrml_factor_at_t).
        coupling = -jnp.sum(gamma_t / 2 * x_t * x_t_minus_1, axis=-1)

        return coupling
    
    def get_gamma(self, t: int) -> float:
        """Get coupling strength at timestep t."""
        return float(self.gamma[t])
    
    def get_full_schedule(self) -> jnp.ndarray:
        """Get full coupling strength schedule."""
        return self.gamma
    
    def to_thrml_factor(
        self,
        nodes_t: List,
        nodes_t_minus_1: List
    ):
        """
        Convert forward coupling to THRML SpinEBMFactor.
        
        Mathematical Note:
        - Forward coupling: E_f = Γ(t)/2 Σ_i x_i^{(t)} x_i^{(t-1)}
        - SpinEBMFactor with two blocks computes: -Σ_{i,j} weights[i,j] * A[i] * B[j]
        - With diagonal weights: -Σ_i weights[i,i] * A[i] * B[i]
        - This gives: -Σ_i (γ/2) * x_i^{(t)} * x_i^{(t-1)} ✅
        
        Args:
            nodes_t: Nodes at timestep t
            nodes_t_minus_1: Nodes at timestep t-1
        
        Returns:
            SpinEBMFactor for timestep t (requires THRML import)
        """
        from thrml.models.discrete_ebm import SpinEBMFactor
        from thrml.block_management import Block
        
        # This method requires a specific timestep t to get gamma_t
        # Since gamma is a schedule, we need to call this for each timestep
        # For now, this is a placeholder showing the structure
        # In practice, you would call this with a specific t
        
        raise NotImplementedError(
            "to_thrml_factor requires a specific timestep t. "
            "Use to_thrml_factor_at_t(t, nodes_t, nodes_t_minus_1) instead."
        )
    
    def to_thrml_factor_at_t(
        self,
        t: int,
        nodes_t: List,
        nodes_t_minus_1: List
    ):
        """
        Convert forward coupling at timestep t to THRML SpinEBMFactor.
        
        Mathematical Note:
        - Forward coupling: E_f = Γ(t)/2 Σ_i x_i^{(t)} x_i^{(t-1)}
        - SpinEBMFactor with two blocks computes: -Σ_{i,j} weights[i,j] * A[i] * B[j]
        - With diagonal weights: -Σ_i weights[i,i] * A[i] * B[i]
        - This gives: -Σ_i (γ/2) * x_i^{(t)} * x_i^{(t-1)} ✅
        
        Args:
            t: Timestep index
            nodes_t: Nodes at timestep t
            nodes_t_minus_1: Nodes at timestep t-1
        
        Returns:
            SpinEBMFactor for timestep t
        """
        from thrml.models.discrete_ebm import SpinEBMFactor
        from thrml.block_management import Block
        
        gamma_t = self.gamma[t]
        n_vars = len(nodes_t)
        
        # Create diagonal weight matrix
        # weights[i, j] = gamma_t/2 if i == j else 0
        weights = jnp.eye(n_vars) * (gamma_t / 2.0)
        
        # SpinEBMFactor with two blocks:
        # - node_groups[0] = Block(nodes_t) = [x_t_0, x_t_1, ..., x_t_{n-1}]
        # - node_groups[1] = Block(nodes_t_minus_1) = [x_{t-1}_0, x_{t-1}_1, ..., x_{t-1}_{n-1}]
        # - weights shape = [n_vars, n_vars]
        # Energy = -Σ_{i,j} weights[i,j] * nodes_t[i] * nodes_t_minus_1[j]
        # With diagonal weights: -Σ_i (γ/2) * x_t_i * x_{t-1}_i ✅
        return SpinEBMFactor(
            node_groups=[Block(nodes_t), Block(nodes_t_minus_1)],
            weights=weights
        )


def test_forward_coupling():
    """Test forward coupling implementation."""
    print("Testing ForwardCoupling...")
    
    config = ForwardCouplingConfig(T=1000, schedule_type="linear")
    coupling = ForwardCoupling(config)
    
    # Test schedule shape
    gamma = coupling.get_full_schedule()
    assert gamma.shape == (1000,), f"Expected shape (1000,), got {gamma.shape}"
    print(f"Schedule shape: {gamma.shape}")
    print(f"Gamma range: [{gamma.min():.4f}, {gamma.max():.4f}]")
    
    # Test coupling energy computation
    x_t = jax.random.randint(jax.random.PRNGKey(0), (4, 64), minval=0, maxval=2) * 2 - 1
    x_t_minus_1 = jax.random.randint(jax.random.PRNGKey(1), (4, 64), minval=0, maxval=2) * 2 - 1
    
    energy = coupling(x_t, x_t_minus_1, t=500)
    assert energy.shape == (4,), f"Expected shape (4,), got {energy.shape}"
    print(f"Coupling energy shape: {energy.shape}")
    print(f"Coupling energy values: {energy}")
    
    # Test cosine schedule
    config_cosine = ForwardCouplingConfig(T=1000, schedule_type="cosine")
    coupling_cosine = ForwardCoupling(config_cosine)
    gamma_cosine = coupling_cosine.get_full_schedule()
    print(f"Cosine schedule range: [{gamma_cosine.min():.4f}, {gamma_cosine.max():.4f}]")
    
    print("[SUCCESS] ForwardCoupling test passed!")


if __name__ == "__main__":
    test_forward_coupling()
