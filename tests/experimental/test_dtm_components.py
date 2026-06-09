"""
Test all DTM components.

Run with: python test_dtm_components.py
"""

import sys
import jax
import jax.numpy as jnp

# Add thermolm_jax to path
sys.path.insert(0, 'd:/individual-research/ThermoLM')

# Import all components
from thermolm_jax.models.quadratic_ebm import QuadraticEBM, QuadraticEBMConfig
from thermolm_jax.models.connectivity import generate_connectivity_pattern
from thermolm_jax.models.forward_coupling import ForwardCoupling, ForwardCouplingConfig
from thermolm_jax.models.latent_graph import SparseGraph, LatentGraphConfig
from thermolm_jax.models.dtm import DTM, DTMConfig
from thermolm_jax.models.thrml_quadratic import THRMLQuadraticEBM
from thermolm_jax.models.binary_autoencoder import BinaryAutoencoder, BinaryAutoencoderConfig
from thermolm_jax.sampling.chromatic_gibbs import chromatic_gibbs_sample
from thermolm_jax.training.contrastive_divergence import improved_contrastive_divergence_loss, CDConfig
from thermolm_jax.training.acp import AdaptiveCorrelationPenalty, ACPConfig, estimate_autocorrelation
from thermolm_jax.training.total_correlation import total_correlation_penalty, TCConfig
from thermolm_jax.training.train_dtm import DTMTrainer, TrainingConfig
from thermolm_jax.training.hybrid_training import train_stage1_autoencoder
from thermolm_jax.evaluation.mixing_time import estimate_mixing_time, MixingTimeConfig
from thermolm_jax.evaluation.energy_landscape import analyze_energy_landscape, EnergyLandscapeConfig
from thermolm_jax.evaluation.tsu_metrics import estimate_tsu_energy_consumption, TSUMetricsConfig


def test_all():
    """Run all component tests."""
    print("=" * 60)
    print("Testing DTM Components")
    print("=" * 60)
    
    key = jax.random.PRNGKey(0)
    
    # Test 1: QuadraticEBM
    print("\n1. Testing QuadraticEBM...")
    config = QuadraticEBMConfig(n_vars=64, beta=1.0)
    ebm = QuadraticEBM(config, key)
    x = jax.random.randint(key, (4, 64), minval=0, maxval=2) * 2 - 1
    energy = ebm(x)
    print(f"   Energy shape: {energy.shape}")
    print(f"   Energy values: {energy}")
    assert energy.shape == (4,)
    print("   [PASSED]")
    
    # Test 2: Connectivity patterns
    print("\n2. Testing connectivity patterns...")
    mask = generate_connectivity_pattern("G16", 64, "bipartite")
    print(f"   Connectivity mask shape: {mask.shape}")
    print(f"   Connectivity density: {jnp.mean(mask):.3f}")
    assert mask.shape == (64, 64)
    print("   [PASSED]")
    
    # Test 3: Forward coupling
    print("\n3. Testing ForwardCoupling...")
    fc_config = ForwardCouplingConfig(T=100, schedule_type="linear")
    fc = ForwardCoupling(fc_config)
    x_t = jax.random.randint(key, (4, 64), minval=0, maxval=2) * 2 - 1
    x_t_minus_1 = jax.random.randint(key, (4, 64), minval=0, maxval=2) * 2 - 1
    coupling = fc(x_t, x_t_minus_1, 50)
    print(f"   Coupling energy shape: {coupling.shape}")
    assert coupling.shape == (4,)
    print("   [PASSED]")
    
    # Test 4: Latent graph
    print("\n4. Testing SparseGraph...")
    lg_config = LatentGraphConfig(n_data_vars=64, n_latent_vars=16)
    graph = SparseGraph(lg_config)
    print(f"   Data nodes: {graph.get_n_data_vars()}")
    print(f"   Latent nodes: {graph.get_n_latent_vars()}")
    print(f"   Total nodes: {graph.get_n_total_vars()}")
    assert graph.get_n_data_vars() == 64
    assert graph.get_n_latent_vars() == 16
    print("   [PASSED]")
    
    # Test 5: DTM
    print("\n5. Testing DTM...")
    dtm_config = DTMConfig(n_data_vars=64, n_latent_vars=16, T=100)
    dtm = DTM(dtm_config, key)
    x_t = dtm.sample_initial_state(4, key)
    x_t_minus_1 = dtm.sample_initial_state(4, key)
    energy = dtm(x_t, x_t_minus_1, 50)
    print(f"   DTM energy shape: {energy.shape}")
    assert energy.shape == (4,)
    print("   [PASSED]")
    
    # Test 6: THRML wrapper
    print("\n6. Testing THRMLQuadraticEBM...")
    J = jax.random.normal(key, (64, 64)) * 0.01
    h = jax.random.normal(key, (64,)) * 0.01
    wrapper = THRMLQuadraticEBM(J, h, 1.0, jnp.ones((64, 64), dtype=bool))
    x = jax.random.randint(key, (4, 64), minval=0, maxval=2) * 2 - 1
    energy = wrapper.compute_energy_from_factors(x, [])
    print(f"   Energy shape: {energy.shape}")
    assert energy.shape == (4,)
    print("   [PASSED]")
    
    # Test 7: Chromatic Gibbs
    print("\n7. Testing chromatic Gibbs sampling...")
    def simple_energy(x):
        return -jnp.sum(x, axis=-1)
    init_state = jax.random.randint(key, (4, 64), minval=0, maxval=2) * 2 - 1
    final_state, info = chromatic_gibbs_sample(simple_energy, init_state, 10, key)
    print(f"   Final state shape: {final_state.shape}")
    assert final_state.shape == init_state.shape
    print("   [PASSED]")
    
    # Test 8: Contrastive divergence
    print("\n8. Testing contrastive divergence...")
    class SimpleModel:
        def __init__(self):
            self.weights = jnp.ones(64)
        def __call__(self, x):
            return -jnp.sum(x * self.weights, axis=-1)
    model = SimpleModel()
    x_data = jax.random.randint(key, (4, 64), minval=0, maxval=2) * 2 - 1
    cd_config = CDConfig(k=1, n_gibbs_steps=10, use_improved_cd=False)
    loss, info = improved_contrastive_divergence_loss(model, x_data, key, cd_config)
    print(f"   CD loss: {loss}")
    print("   [PASSED]")
    
    # Test 9: ACP
    print("\n9. Testing ACP...")
    acp_config = ACPConfig(target_autocorr=0.03, lag=10)
    acp = AdaptiveCorrelationPenalty(acp_config)
    samples = jax.random.randint(key, (100, 64), minval=0, maxval=2) * 2 - 1
    autocorr = estimate_autocorrelation(samples, lag=10)
    acp = acp.update(samples)
    lambda_new = acp.get_lambda()
    print(f"   Autocorrelation: {autocorr:.4f}")
    print(f"   Lambda: {lambda_new:.6f}")
    print("   [PASSED]")
    
    # Test 10: Total correlation
    print("\n10. Testing total correlation...")
    tc_config = TCConfig(lambda_tc=0.1)
    tc_penalty = total_correlation_penalty(samples, tc_config)
    print(f"   TC penalty: {tc_penalty:.4f}")
    print("   [PASSED]")
    
    # Test 11: Mixing time
    print("\n11. Testing mixing time...")
    mix_config = MixingTimeConfig(max_steps=100, lag=10)
    init_state = jax.random.randint(key, (4, 64), minval=0, maxval=2) * 2 - 1
    metrics = estimate_mixing_time(simple_energy, init_state, mix_config, key)
    print(f"   Mixing time: {metrics['mixing_time']}")
    print("   [PASSED]")
    
    # Test 12: Energy landscape
    print("\n12. Testing energy landscape...")
    el_config = EnergyLandscapeConfig(n_samples=100, n_bins=20)
    metrics = analyze_energy_landscape(simple_energy, samples, el_config)
    print(f"   Mean energy: {metrics['mean_energy']:.4f}")
    print(f"   Std energy: {metrics['std_energy']:.4f}")
    print("   [PASSED]")
    
    # Test 13: TSU metrics
    print("\n13. Testing TSU metrics...")
    tsu_config = TSUMetricsConfig()
    energy_metrics = estimate_tsu_energy_consumption(1024, "G16", 1000, 100, tsu_config)
    print(f"   Total energy: {energy_metrics['total_energy_joules']:.2e} J")
    print("   [PASSED]")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_all()
