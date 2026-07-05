import shutil
import os

moves = [
    ("thermolm_jax/models/discrete_edlm.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/models/discrete_energy.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/models/sampler.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/models/d3pm.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/models/fsq.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/models/hybrid_edlm.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/models/hybrid_energy.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/models/continuous_encoder.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/models/quantization.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/models/binary_autoencoder.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/models/edlm.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/models/energy_function.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/models/thrml_discrete.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/models/adaln.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/models/dit_block.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/models/rotary.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/models/timestep.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/models/dtm.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/models/latent_graph.py", "experimental/thermolm_jax/models/"),
    ("thermolm_jax/training/train_discrete_edlm.py", "experimental/thermolm_jax/training/"),
    ("thermolm_jax/training/train_hybrid_edlm.py", "experimental/thermolm_jax/training/"),
    ("thermolm_jax/training/thrml_training.py", "experimental/thermolm_jax/training/"),
    ("thermolm_jax/training/thrml_flax_coexistence.py", "experimental/thermolm_jax/training/"),
    ("thermolm_jax/training/acp.py", "experimental/thermolm_jax/training/"),
    ("thermolm_jax/training/total_correlation.py", "experimental/thermolm_jax/training/"),
    ("thermolm_jax/training/hybrid_training.py", "experimental/thermolm_jax/training/"),
    ("thermolm_jax/training/distributed.py", "experimental/thermolm_jax/training/"),
    ("thermolm_jax/training/train_dtm.py", "experimental/thermolm_jax/training/"),
    ("thermolm_jax/training/ema.py", "experimental/thermolm_jax/training/"),
    ("thermolm_jax/training/base_trainer.py", "experimental/thermolm_jax/training/"),
    ("thermolm_jax/training/checkpoint.py", "experimental/thermolm_jax/training/"),
    ("thermolm_jax/training/optimizer.py", "experimental/thermolm_jax/training/"),
    ("thermolm_jax/training/scheduler.py", "experimental/thermolm_jax/training/"),
    ("thermolm_jax/data/base_loader.py", "experimental/thermolm_jax/data/"),
    ("thermolm_jax/data/preprocessing.py", "experimental/thermolm_jax/data/"),
    ("thermolm_jax/data/tokenizers.py", "experimental/thermolm_jax/data/"),
    ("thermolm_jax/data/wikitext_jax.py", "experimental/thermolm_jax/data/"),
    ("thermolm_jax/evaluation/analysis.py", "experimental/thermolm_jax/evaluation/"),
    ("thermolm_jax/evaluation/benchmark.py", "experimental/thermolm_jax/evaluation/"),
    ("thermolm_jax/evaluation/compare.py", "experimental/thermolm_jax/evaluation/"),
    ("thermolm_jax/evaluation/energy_landscape.py", "experimental/thermolm_jax/evaluation/"),
    ("thermolm_jax/evaluation/metrics.py", "experimental/thermolm_jax/evaluation/"),
    ("thermolm_jax/evaluation/mixing_time.py", "experimental/thermolm_jax/evaluation/"),
    ("thermolm_jax/evaluation/tsu_metrics.py", "experimental/thermolm_jax/evaluation/"),
    ("scripts/compare_models.py", "experimental/"),
    ("data/wikitext_loader.py", "experimental/"),
]

for src, dst in moves:
    if os.path.exists(src):
        os.makedirs(dst, exist_ok=True)
        shutil.move(src, dst)
        print(f"Moved {src} -> {dst}")
    else:
        print(f"SKIP (not found): {src}")

print("Done.")
