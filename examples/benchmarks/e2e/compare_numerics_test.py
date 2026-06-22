# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test to compare numerics between PyTorch and TorchAX on TPU."""

import os
import math
import subprocess
import sys
from absl import app
from absl import flags
from absl.testing import absltest
import torch
from examples.gemma4 import tpu_attention  # pylint: disable=unused-import
from tests import module_registry
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm, apply_rotary_pos_emb, Gemma4TextMLP

FLAGS = flags.FLAGS
DEBUG = os.environ.get("TORCH_TPU_DEBUG", "False").lower() == "true"

flags.DEFINE_enum(
    "run_mode",
    "runner",
    [
        "runner",
        "pytorch",
        "torchax",
        "kernel_pytorch",
        "kernel_jax",
        "full_attn_pytorch",
        "full_attn_jax",
        "rmsnorm_pytorch",
        "rmsnorm_jax",
        "rope_pytorch",
        "rope_jax",
        "mlp_pytorch",
        "mlp_jax",
        "decoder_pytorch",
        "decoder_jax",
        "splash_pytorch",
    ],
    "Execution mode.",
)
flags.DEFINE_integer("layer_idx", 0, "Layer index to test in isolated tests.")
flags.DEFINE_string("model_name", "google/gemma-3-270m", "Model name.")
flags.DEFINE_integer("seq_len", 128, "Sequence length.")
flags.DEFINE_integer("batch_size", 2, "Batch size.")
flags.DEFINE_bool("skip_swa", False, "Force skipping SWA in config.")

# Override default weights_base_path to avoid remote storage access hang in remote test runner
flags.FLAGS.set_default("weights_base_path", "")


def get_tmp_dir():
  return os.environ.get("TEST_TMPDIR", "/tmp")


def get_shrunken_module_spec(
    model_name, attn_implementation="torch_tpu_gemma4"
):
  registry = module_registry.ModuleRegistry()

  # Print IDs to check singleton
  from transformers import modeling_utils as mu
  from transformers.models.gemma4 import modeling_gemma4 as mg

  print(
      f"[DEBUG_ID] mu.ALL_ATTENTION_FUNCTIONS: {id(mu.ALL_ATTENTION_FUNCTIONS)}"
  )
  print(f"[DEBUG_ID] mu file: {mu.__file__}")
  print(f"[DEBUG_ID] mu name: {mu.__name__}")
  print(
      f"[DEBUG_ID] mg.ALL_ATTENTION_FUNCTIONS: {id(mg.ALL_ATTENTION_FUNCTIONS)}"
  )
  print(
      "[DEBUG_ID] registered keys:"
      f" {list(mu.ALL_ATTENTION_FUNCTIONS.valid_keys())}"
  )

  def modify_config(config):

    print(f"[DEBUG_CONFIG] model_type: {getattr(config, 'model_type', None)}")
    if getattr(config, "model_type", None) == "gemma4":
      print(
          "[DEBUG_CONFIG] Setting _attn_implementation to"
          f" {attn_implementation}"
      )
      config._attn_implementation = attn_implementation
      if FLAGS.skip_swa:
        print(
            f"[DEBUG_CONFIG] Force setting use_bidirectional_attention to"
            f" 'vision' (skip SWA)"
        )
        config.text_config.use_bidirectional_attention = "vision"
    return config

  return registry.get_module_spec(
      "transformers",
      model_name,
      load_weights=False,
      modify_config_hook=modify_config,
  )


class CompareNumericsTest(absltest.TestCase):

  def test_splash_vs_ref_swa(self):
    tmp_dir = get_tmp_dir()
    batch_size = 2
    n_heads = 8
    n_kv_heads = 1
    seq_len = 128
    head_dim = 256
    window_size = 64

    # Create random inputs on CPU
    q = torch.randn(
        batch_size, n_heads, seq_len, head_dim, dtype=torch.bfloat16
    )
    k = torch.randn(
        batch_size, n_kv_heads, seq_len, head_dim, dtype=torch.bfloat16
    )
    v = torch.randn(
        batch_size, n_kv_heads, seq_len, head_dim, dtype=torch.bfloat16
    )

    inputs = {
        "q": q,
        "k": k,
        "v": v,
        "window_size": window_size,
    }
    torch.save(inputs, os.path.join(tmp_dir, "splash_vs_ref_inputs.pt"))

    # Run PyTorch Splash Subprocess (TPU)
    sub_env = os.environ.copy()
    sub_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
    cmd = [
        sys.argv[0],
        "--run_mode=splash_pytorch",
        f"--model_name={FLAGS.model_name}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=sub_env)
    print("STDOUT (Splash PyTorch):")
    print(res.stdout)
    print("STDERR (Splash PyTorch):")
    print(res.stderr)
    self.assertEqual(res.returncode, 0, "Splash PyTorch subprocess failed")

    # Load Splash output
    out_splash = torch.load(os.path.join(tmp_dir, "out_splash_pt.pt"))

    # Run Reference SWA on CPU (float32 for reference)
    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()

    # Generate mask (True means keep)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    window_mask = torch.tril(
        torch.ones(seq_len, seq_len), diagonal=-window_size
    ).bool()
    mask = ~(causal_mask | window_mask)

    out_ref = custom_sdpa_upcast(
        q_f32,
        k_f32,
        v_f32,
        attn_mask=mask,
        scale=1.0 / math.sqrt(head_dim),
        enable_gqa=True,
    )

    out_ref_bf16 = out_ref.bfloat16()

    diff = (out_splash - out_ref_bf16).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    flat_splash = out_splash.flatten().to(torch.float32)
    flat_ref = out_ref_bf16.flatten().to(torch.float32)
    cos_sim = torch.nn.functional.cosine_similarity(
        flat_splash, flat_ref, dim=0
    ).item()

    print(f"\n--- Splash vs Reference SWA Results ---")
    print(f"Splash    OUT | {tpu_attention.get_stats_str(out_splash)}")
    print(f"Reference OUT | {tpu_attention.get_stats_str(out_ref_bf16)}")
    print(f"Max Absolute Difference:  {max_diff:.6f}")
    print(f"Mean Absolute Difference: {mean_diff:.6f}")
    print(f"Cosine Similarity:        {cos_sim:.6f}")
    print(f"---------------------------------------\n")

    self.assertGreater(
        cos_sim, 0.99, "Splash vs Reference SWA cosine similarity too low"
    )

  def test_numerics_match(self):
    if FLAGS.run_mode != "runner":
      return

    # 1. Generate data
    print("Generating initial weights and inputs on CPU...")

    module_spec = get_shrunken_module_spec(FLAGS.model_name)

    # Initialize model on CPU
    model = module_spec.module_factory().to(torch.bfloat16)
    print(
        "[DEBUG_RUN] model config _attn_implementation:"
        f" {getattr(model.config, '_attn_implementation', None)}"
    )
    config = model.config
    tmp_dir = get_tmp_dir()
    weights_path = os.path.join(tmp_dir, "weights.pt")
    inputs_path = os.path.join(tmp_dir, "inputs.pt")

    torch.save(model.state_dict(), weights_path)

    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is None and hasattr(config, "text_config"):
      vocab_size = getattr(config.text_config, "vocab_size", None)
    if vocab_size is None:
      vocab_size = 32000
    input_ids = torch.randint(
        0, vocab_size, (FLAGS.batch_size, FLAGS.seq_len), dtype=torch.int64
    )
    inputs = {"input_ids": input_ids}
    torch.save(inputs, inputs_path)

    # 2. Run PyTorch in subprocess
    print("Spawning PyTorch subprocess...")
    sub_env_py = os.environ.copy()
    sub_env_py["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"

    py_cmd = [
        sys.argv[0],
        "--run_mode=pytorch",
        f"--model_name={FLAGS.model_name}",
        f"--seq_len={FLAGS.seq_len}",
        f"--batch_size={FLAGS.batch_size}",
    ]
    res_py = subprocess.run(
        py_cmd, capture_output=True, text=True, env=sub_env_py
    )
    print("STDOUT (PyTorch):")
    print(res_py.stdout)
    print("STDERR (PyTorch):")
    print(res_py.stderr)
    self.assertEqual(res_py.returncode, 0, "PyTorch subprocess failed")

    # 3. Run TorchAX in subprocess
    print("Spawning TorchAX subprocess...")
    sub_env_ax = os.environ.copy()
    sub_env_ax["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

    ax_cmd = [
        sys.argv[0],
        "--run_mode=torchax",
        f"--model_name={FLAGS.model_name}",
        f"--seq_len={FLAGS.seq_len}",
        f"--batch_size={FLAGS.batch_size}",
    ]
    res_ax = subprocess.run(
        ax_cmd, capture_output=True, text=True, env=sub_env_ax
    )
    print("STDOUT (TorchAX):")
    print(res_ax.stdout)
    print("STDERR (TorchAX):")
    print(res_ax.stderr)
    self.assertEqual(res_ax.returncode, 0, "TorchAX subprocess failed")

    # 4. Compare
    pytorch_out_path = os.path.join(tmp_dir, "pytorch_out.pt")
    torchax_out_path = os.path.join(tmp_dir, "torchax_out.pt")

    py_logits = torch.load(pytorch_out_path).to(torch.float32)
    ax_logits = torch.load(torchax_out_path).to(torch.float32)

    print(f"PyTorch Logits shape: {py_logits.shape}")
    print(f"TorchAX Logits shape: {ax_logits.shape}")

    self.assertEqual(py_logits.shape, ax_logits.shape, "Shapes do not match")

    diff = torch.abs(py_logits - ax_logits)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()

    cos = torch.nn.CosineSimilarity(dim=-1)
    similarity = cos(
        py_logits.reshape(-1, py_logits.shape[-1]),
        ax_logits.reshape(-1, ax_logits.shape[-1]),
    )
    mean_sim = torch.mean(similarity).item()

    print("\n--- Numerical Comparison Results ---")
    print(f"Max Absolute Difference:  {max_diff:.6f}")
    print(f"Mean Absolute Difference: {mean_diff:.6f}")
    print(f"Mean Cosine Similarity:   {mean_sim:.6f}")
    print("------------------------------------\n")

    # Assert similarity is high
    self.assertGreater(mean_sim, 0.999, "Cosine similarity too low")
    # Assert max diff is reasonable for bfloat16
    self.assertLess(max_diff, 1e-1, "Max difference too high")

  def test_kernel_parity(self):
    tmp_dir = get_tmp_dir()
    q = torch.randn(2, 8, 1024, 256, dtype=torch.bfloat16)
    k = torch.randn(2, 1, 1024, 256, dtype=torch.bfloat16)
    v = torch.randn(2, 1, 1024, 256, dtype=torch.bfloat16)
    inputs = {
        "q": q,
        "k": k,
        "v": v,
        "scale": 1.0,
        "local_window_size": 511,
        "enable_gqa": True,
    }
    torch.save(inputs, os.path.join(tmp_dir, "kernel_inputs.pt"))

    sub_env_py = os.environ.copy()
    sub_env_py["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
    cmd_py = [
        sys.argv[0],
        "--run_mode=kernel_pytorch",
        f"--model_name={FLAGS.model_name}",
    ]
    res_py = subprocess.run(
        cmd_py, capture_output=True, text=True, env=sub_env_py
    )
    print("STDOUT (Kernel PyTorch):")
    print(res_py.stdout)
    print("STDERR (Kernel PyTorch):")
    print(res_py.stderr)
    self.assertEqual(res_py.returncode, 0, "Kernel PyTorch subprocess failed")

    sub_env_ax = os.environ.copy()
    sub_env_ax["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    cmd_jax = [
        sys.argv[0],
        "--run_mode=kernel_jax",
        f"--model_name={FLAGS.model_name}",
    ]
    res_jax = subprocess.run(
        cmd_jax, capture_output=True, text=True, env=sub_env_ax
    )
    print("STDOUT (Kernel JAX):")
    print(res_jax.stdout)
    print("STDERR (Kernel JAX):")
    print(res_jax.stderr)
    self.assertEqual(res_jax.returncode, 0, "Kernel JAX subprocess failed")

    out_pt = torch.load(os.path.join(tmp_dir, "out_kernel_pt.pt"))
    out_jax = torch.load(os.path.join(tmp_dir, "out_kernel_jax.pt"))

    diff = (out_pt - out_jax).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    flat_pt = out_pt.flatten().to(torch.float32)
    flat_jax = out_jax.flatten().to(torch.float32)
    cos_sim = torch.nn.functional.cosine_similarity(
        flat_pt, flat_jax, dim=0
    ).item()

    print(f"\n--- Kernel Comparison Results ---")
    print(f"Max Absolute Difference:  {max_diff:.6f}")
    print(f"Mean Absolute Difference: {mean_diff:.6f}")
    print(f"Cosine Similarity:        {cos_sim:.6f}")
    print(f"----------------------------------\n")

    self.assertGreater(cos_sim, 0.999, "Kernel cosine similarity too low")

  def test_full_attn_parity(self):
    tmp_dir = get_tmp_dir()

    # Gemma 4 head dim is 128
    batch_size = 2
    n_heads = 8
    seq_len = 512
    head_dim = 128

    # Create random inputs on CPU
    q = torch.randn(
        batch_size, n_heads, seq_len, head_dim, dtype=torch.bfloat16
    )
    k = torch.randn(
        batch_size, n_heads, seq_len, head_dim, dtype=torch.bfloat16
    )
    v = torch.randn(
        batch_size, n_heads, seq_len, head_dim, dtype=torch.bfloat16
    )

    inputs = {
        "q": q,
        "k": k,
        "v": v,
    }
    torch.save(inputs, os.path.join(tmp_dir, "full_attn_inputs.pt"))

    # Run PyTorch Subprocess
    sub_env_py = os.environ.copy()
    sub_env_py["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
    cmd_py = [
        sys.argv[0],
        "--run_mode=full_attn_pytorch",
        f"--model_name={FLAGS.model_name}",
    ]
    res_py = subprocess.run(
        cmd_py, capture_output=True, text=True, env=sub_env_py
    )
    print("STDOUT (Full Attn PyTorch):")
    print(res_py.stdout)
    print("STDERR (Full Attn PyTorch):")
    print(res_py.stderr)
    self.assertEqual(
        res_py.returncode, 0, "Full Attn PyTorch subprocess failed"
    )

    # Run JAX Subprocess
    sub_env_ax = os.environ.copy()
    sub_env_ax["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    cmd_jax = [
        sys.argv[0],
        "--run_mode=full_attn_jax",
        f"--model_name={FLAGS.model_name}",
    ]
    res_jax = subprocess.run(
        cmd_jax, capture_output=True, text=True, env=sub_env_ax
    )
    print("STDOUT (Full Attn JAX):")
    print(res_jax.stdout)
    print("STDERR (Full Attn JAX):")
    print(res_jax.stderr)
    self.assertEqual(res_jax.returncode, 0, "Full Attn JAX subprocess failed")

    # Load and compare
    out_pt = torch.load(os.path.join(tmp_dir, "out_full_attn_pt.pt"))
    out_jax = torch.load(os.path.join(tmp_dir, "out_full_attn_jax.pt"))

    diff = (out_pt - out_jax).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    flat_pt = out_pt.flatten().to(torch.float32)
    flat_jax = out_jax.flatten().to(torch.float32)
    cos_sim = torch.nn.functional.cosine_similarity(
        flat_pt, flat_jax, dim=0
    ).item()

    print(f"\n--- Full Attention Parity Results ---")
    print(f"Max Absolute Difference:  {max_diff:.6f}")
    print(f"Mean Absolute Difference: {mean_diff:.6f}")
    print(f"Cosine Similarity:        {cos_sim:.6f}")
    print(f"-------------------------------------\n")

    self.assertGreater(
        cos_sim, 0.999, "Full Attention cosine similarity too low"
    )

  def test_rmsnorm_parity(self):
    tmp_dir = get_tmp_dir()
    dim = 512
    x = torch.randn(2, 128, dim, dtype=torch.bfloat16)
    weight = torch.randn(dim, dtype=torch.bfloat16)

    inputs = {
        "x": x,
        "weight": weight,
    }
    torch.save(inputs, os.path.join(tmp_dir, "rmsnorm_inputs.pt"))

    # Run PyTorch Subprocess
    sub_env_py = os.environ.copy()
    sub_env_py["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
    cmd_py = [
        sys.argv[0],
        "--run_mode=rmsnorm_pytorch",
    ]
    res_py = subprocess.run(
        cmd_py, capture_output=True, text=True, env=sub_env_py
    )
    print("STDOUT (RMSNorm PyTorch):")
    print(res_py.stdout)
    print("STDERR (RMSNorm PyTorch):")
    print(res_py.stderr)
    self.assertEqual(res_py.returncode, 0, "RMSNorm PyTorch subprocess failed")

    # Run JAX Subprocess
    sub_env_ax = os.environ.copy()
    sub_env_ax["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    cmd_jax = [
        sys.argv[0],
        "--run_mode=rmsnorm_jax",
    ]
    res_jax = subprocess.run(
        cmd_jax, capture_output=True, text=True, env=sub_env_ax
    )
    print("STDOUT (RMSNorm JAX):")
    print(res_jax.stdout)
    print("STDERR (RMSNorm JAX):")
    print(res_jax.stderr)
    self.assertEqual(res_jax.returncode, 0, "RMSNorm JAX subprocess failed")

    # Load and compare
    out_pt = torch.load(os.path.join(tmp_dir, "out_rmsnorm_pt.pt"))
    out_jax = torch.load(os.path.join(tmp_dir, "out_rmsnorm_jax.pt"))

    diff = (out_pt - out_jax).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    flat_pt = out_pt.flatten().to(torch.float32)
    flat_jax = out_jax.flatten().to(torch.float32)
    cos_sim = torch.nn.functional.cosine_similarity(
        flat_pt, flat_jax, dim=0
    ).item()

    print(f"\n--- RMSNorm Parity Results ---")
    print(f"Max Absolute Difference:  {max_diff:.6f}")
    print(f"Mean Absolute Difference: {mean_diff:.6f}")
    print(f"Cosine Similarity:        {cos_sim:.6f}")
    print(f"-------------------------------\n")

    self.assertGreater(cos_sim, 0.999, "RMSNorm cosine similarity too low")

  def test_rope_parity(self):
    tmp_dir = get_tmp_dir()
    batch_size = 2
    n_heads = 8
    seq_len = 128
    head_dim = 128

    x = torch.randn(
        batch_size, seq_len, n_heads, head_dim, dtype=torch.bfloat16
    )
    cos = torch.randn(batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    sin = torch.randn(batch_size, seq_len, head_dim, dtype=torch.bfloat16)

    inputs = {
        "x": x,
        "cos": cos,
        "sin": sin,
    }
    torch.save(inputs, os.path.join(tmp_dir, "rope_inputs.pt"))

    # Run PyTorch Subprocess
    sub_env_py = os.environ.copy()
    sub_env_py["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
    cmd_py = [
        sys.argv[0],
        "--run_mode=rope_pytorch",
    ]
    res_py = subprocess.run(
        cmd_py, capture_output=True, text=True, env=sub_env_py
    )
    print("STDOUT (RoPE PyTorch):")
    print(res_py.stdout)
    print("STDERR (RoPE PyTorch):")
    print(res_py.stderr)
    self.assertEqual(res_py.returncode, 0, "RoPE PyTorch subprocess failed")

    # Run JAX Subprocess
    sub_env_ax = os.environ.copy()
    sub_env_ax["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    cmd_jax = [
        sys.argv[0],
        "--run_mode=rope_jax",
    ]
    res_jax = subprocess.run(
        cmd_jax, capture_output=True, text=True, env=sub_env_ax
    )
    print("STDOUT (RoPE JAX):")
    print(res_jax.stdout)
    print("STDERR (RoPE JAX):")
    print(res_jax.stderr)
    self.assertEqual(res_jax.returncode, 0, "RoPE JAX subprocess failed")

    # Load and compare
    out_pt = torch.load(os.path.join(tmp_dir, "out_rope_pt.pt"))
    out_jax = torch.load(os.path.join(tmp_dir, "out_rope_jax.pt"))

    diff = (out_pt - out_jax).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    flat_pt = out_pt.flatten().to(torch.float32)
    flat_jax = out_jax.flatten().to(torch.float32)
    cos_sim = torch.nn.functional.cosine_similarity(
        flat_pt, flat_jax, dim=0
    ).item()

    print(f"\n--- RoPE Parity Results ---")
    print(f"Max Absolute Difference:  {max_diff:.6f}")
    print(f"Mean Absolute Difference: {mean_diff:.6f}")
    print(f"Cosine Similarity:        {cos_sim:.6f}")
    print(f"----------------------------\n")

    self.assertGreater(cos_sim, 0.999, "RoPE cosine similarity too low")

  def test_mlp_parity(self):
    tmp_dir = get_tmp_dir()

    module_spec = get_shrunken_module_spec(FLAGS.model_name)
    config = module_spec.config
    if not hasattr(config, "text_config"):
      print(
          "Skipping MLP parity test for non-multimodal model:"
          f" {FLAGS.model_name}"
      )
      return
    config = config.text_config

    x = torch.randn(2, 128, config.hidden_size, dtype=torch.bfloat16)

    std_proj = 0.02
    gate_weight = (
        torch.randn(
            config.intermediate_size, config.hidden_size, dtype=torch.bfloat16
        )
        * std_proj
    )
    up_weight = (
        torch.randn(
            config.intermediate_size, config.hidden_size, dtype=torch.bfloat16
        )
        * std_proj
    )
    down_weight = (
        torch.randn(
            config.hidden_size, config.intermediate_size, dtype=torch.bfloat16
        )
        * std_proj
    )

    inputs = {
        "x": x,
        "gate_weight": gate_weight,
        "up_weight": up_weight,
        "down_weight": down_weight,
    }
    torch.save(inputs, os.path.join(tmp_dir, "mlp_inputs.pt"))

    # Run PyTorch Subprocess
    sub_env_py = os.environ.copy()
    sub_env_py["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
    cmd_py = [
        sys.argv[0],
        "--run_mode=mlp_pytorch",
        f"--model_name={FLAGS.model_name}",
    ]
    res_py = subprocess.run(
        cmd_py, capture_output=True, text=True, env=sub_env_py
    )
    print("STDOUT (MLP PyTorch):")
    print(res_py.stdout)
    print("STDERR (MLP PyTorch):")
    print(res_py.stderr)
    self.assertEqual(res_py.returncode, 0, "MLP PyTorch subprocess failed")

    # Run JAX Subprocess
    sub_env_ax = os.environ.copy()
    sub_env_ax["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    cmd_jax = [
        sys.argv[0],
        "--run_mode=mlp_jax",
        f"--model_name={FLAGS.model_name}",
    ]
    res_jax = subprocess.run(
        cmd_jax, capture_output=True, text=True, env=sub_env_ax
    )
    print("STDOUT (MLP JAX):")
    print(res_jax.stdout)
    print("STDERR (MLP JAX):")
    print(res_jax.stderr)
    self.assertEqual(res_jax.returncode, 0, "MLP JAX subprocess failed")

    # Load and compare
    out_pt = torch.load(os.path.join(tmp_dir, "out_mlp_pt.pt"))
    out_jax = torch.load(os.path.join(tmp_dir, "out_mlp_jax.pt"))

    diff = (out_pt - out_jax).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    flat_pt = out_pt.flatten().to(torch.float32)
    flat_jax = out_jax.flatten().to(torch.float32)
    cos_sim = torch.nn.functional.cosine_similarity(
        flat_pt, flat_jax, dim=0
    ).item()

    print(f"\n--- MLP Parity Results ---")
    print(
        f"PyTorch OUT | min: {out_pt.min().item():.4f}, max:"
        f" {out_pt.max().item():.4f}, mean: {out_pt.mean().item():.4f}"
    )
    print(
        f"JAX     OUT | min: {out_jax.min().item():.4f}, max:"
        f" {out_jax.max().item():.4f}, mean: {out_jax.mean().item():.4f}"
    )
    print(f"Max Absolute Difference:  {max_diff:.6f}")
    print(f"Mean Absolute Difference: {mean_diff:.6f}")
    print(f"Cosine Similarity:        {cos_sim:.6f}")
    print(f"---------------------------\n")

    self.assertGreater(cos_sim, 0.999, "MLP cosine similarity too low")

  def test_decoder_layer_parity_swa_active(self):
    FLAGS.skip_swa = False
    self._run_decoder_layer_parity_test()

  def test_decoder_layer_parity_swa_skipped(self):
    FLAGS.skip_swa = True
    self._run_decoder_layer_parity_test()

  def _run_decoder_layer_parity_test(self):
    tmp_dir = get_tmp_dir()

    module_spec = get_shrunken_module_spec(FLAGS.model_name)
    config = module_spec.config
    if not hasattr(config, "text_config"):
      print(
          "Skipping Decoder Layer parity test for non-multimodal model:"
          f" {FLAGS.model_name}"
      )
      return

    model_cpu = module_spec.module_factory().to(torch.bfloat16)
    torch.save(
        model_cpu.state_dict(), os.path.join(tmp_dir, "decoder_weights.pt")
    )

    if FLAGS.skip_swa:
      layers_to_test = [0]
    else:
      layers_to_test = [0, 4]

    for layer_idx in layers_to_test:
      print(
          f"\n================ Running Decoder Layer {layer_idx} Parity"
          " ================"
      )
      hidden_states = torch.randn(
          FLAGS.batch_size,
          FLAGS.seq_len,
          config.text_config.hidden_size,
          dtype=torch.bfloat16,
      )
      per_layer_input = torch.randn(
          FLAGS.batch_size,
          FLAGS.seq_len,
          config.text_config.hidden_size_per_layer_input,
          dtype=torch.bfloat16,
      )

      is_sliding = (
          config.text_config.layer_types[layer_idx] == "sliding_attention"
      )
      head_dim = (
          config.text_config.global_head_dim
          if not is_sliding and config.text_config.global_head_dim
          else config.text_config.head_dim
      )
      cos = torch.randn(1, FLAGS.seq_len, head_dim, dtype=torch.bfloat16)
      sin = torch.randn(1, FLAGS.seq_len, head_dim, dtype=torch.bfloat16)

      first_kv_shared_layer_idx = (
          config.text_config.num_hidden_layers
          - getattr(config.text_config, "num_kv_shared_layers", 0)
      )
      is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx >= 0

      attention_mask = None
      if is_sliding and not FLAGS.skip_swa:
        causal_mask = torch.triu(
            torch.ones(FLAGS.seq_len, FLAGS.seq_len), diagonal=1
        ).bool()
        window_mask = torch.tril(
            torch.ones(FLAGS.seq_len, FLAGS.seq_len),
            diagonal=-config.text_config.sliding_window,
        ).bool()
        attention_mask = ~(causal_mask | window_mask)
        attention_mask = (
            attention_mask.unsqueeze(0)
            .unsqueeze(0)
            .expand(FLAGS.batch_size, 1, -1, -1)
        )

      inputs = {
          "hidden_states": hidden_states,
          "per_layer_input": per_layer_input,
          "cos": cos,
          "sin": sin,
          "attention_mask": attention_mask,
      }

      if is_kv_shared_layer:
        use_alternative_attention = (
            getattr(config.text_config, "attention_k_eq_v", False)
            and not is_sliding
        )
        num_key_value_heads = (
            getattr(config.text_config, "num_global_key_value_heads", 1)
            if use_alternative_attention
            else getattr(config.text_config, "num_key_value_heads", 1)
        )
        kv_shape = (
            FLAGS.batch_size,
            num_key_value_heads,
            FLAGS.seq_len,
            head_dim,
        )
        dummy_k = torch.randn(kv_shape, dtype=torch.bfloat16)
        dummy_v = torch.randn(kv_shape, dtype=torch.bfloat16)
        layer_type = config.text_config.layer_types[layer_idx]
        inputs["shared_kv_states"] = {layer_type: (dummy_k, dummy_v)}

      torch.save(
          inputs, os.path.join(tmp_dir, f"decoder_inputs_{layer_idx}.pt")
      )

      # Run PyTorch Subprocess
      sub_env_py = os.environ.copy()
      sub_env_py["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
      cmd_py = [
          sys.argv[0],
          "--run_mode=decoder_pytorch",
          f"--model_name={FLAGS.model_name}",
          f"--layer_idx={layer_idx}",
          f"--seq_len={FLAGS.seq_len}",
          f"--batch_size={FLAGS.batch_size}",
          f"--skip_swa={FLAGS.skip_swa}",
      ]
      res_py = subprocess.run(
          cmd_py, capture_output=True, text=True, env=sub_env_py
      )
      print(f"STDOUT (Decoder PyTorch L{layer_idx}):")
      print(res_py.stdout)
      print(f"STDERR (Decoder PyTorch L{layer_idx}):")
      print(res_py.stderr)
      self.assertEqual(
          res_py.returncode,
          0,
          f"Decoder PyTorch L{layer_idx} subprocess failed",
      )

      # Run JAX Subprocess
      sub_env_ax = os.environ.copy()
      sub_env_ax["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
      cmd_jax = [
          sys.argv[0],
          "--run_mode=decoder_jax",
          f"--model_name={FLAGS.model_name}",
          f"--layer_idx={layer_idx}",
          f"--seq_len={FLAGS.seq_len}",
          f"--batch_size={FLAGS.batch_size}",
          f"--skip_swa={FLAGS.skip_swa}",
      ]
      res_jax = subprocess.run(
          cmd_jax, capture_output=True, text=True, env=sub_env_ax
      )
      print(f"STDOUT (Decoder JAX L{layer_idx}):")
      print(res_jax.stdout)
      print(f"STDERR (Decoder JAX L{layer_idx}):")
      print(res_jax.stderr)
      self.assertEqual(
          res_jax.returncode, 0, f"Decoder JAX L{layer_idx} subprocess failed"
      )

      # Load and compare
      out_pt = torch.load(
          os.path.join(tmp_dir, f"out_decoder_pt_{layer_idx}.pt")
      )
      out_jax = torch.load(
          os.path.join(tmp_dir, f"out_decoder_jax_{layer_idx}.pt")
      )

      diff = (out_pt - out_jax).abs()
      max_diff = diff.max().item()
      mean_diff = diff.mean().item()

      flat_pt = out_pt.flatten().to(torch.float32)
      flat_jax = out_jax.flatten().to(torch.float32)
      cos_sim = torch.nn.functional.cosine_similarity(
          flat_pt, flat_jax, dim=0
      ).item()

      print(f"\n--- Decoder Layer {layer_idx} Parity Results ---")
      print(
          f"PyTorch OUT | min: {out_pt.min().item():.4f}, max:"
          f" {out_pt.max().item():.4f}, mean: {out_pt.mean().item():.4f}"
      )
      print(
          f"JAX     OUT | min: {out_jax.min().item():.4f}, max:"
          f" {out_jax.max().item():.4f}, mean: {out_jax.mean().item():.4f}"
      )
      print(f"Max Absolute Difference:  {max_diff:.6f}")
      print(f"Mean Absolute Difference: {mean_diff:.6f}")
      print(f"Cosine Similarity:        {cos_sim:.6f}")
      print(f"-----------------------------------------\n")

      self.assertGreater(
          cos_sim, 0.999, f"Decoder L{layer_idx} cosine similarity too low"
      )


def run_pytorch_sub():
  # Force load torch_tpu backend
  from torch_tpu import _loader

  _loader.load()
  import torch.nn.functional as F

  F.scaled_dot_product_attention = custom_sdpa_upcast
  tmp_dir = get_tmp_dir()
  device = torch.device("tpu")

  module_spec = get_shrunken_module_spec(FLAGS.model_name)
  model = module_spec.module_factory().to(torch.float32)
  print(
      "[DEBUG_RUN_PYTORCH] model config _attn_implementation:"
      f" {getattr(model.config, '_attn_implementation', None)}"
  )
  if hasattr(model, "model") and hasattr(model.model, "language_model"):
    print(
        "[DEBUG_RUN_PYTORCH] model.model config _attn_implementation:"
        f" {getattr(model.model.config, '_attn_implementation', None)}"
    )
    print(
        "[DEBUG_RUN_PYTORCH] model.model.language_model config"
        " _attn_implementation:"
        f" {getattr(model.model.language_model.config, '_attn_implementation', None)}"
    )
    if len(model.model.language_model.layers) > 0:
      print(
          "[DEBUG_RUN_PYTORCH] layer 0 config _attn_implementation:"
          f" {getattr(model.model.language_model.layers[0].self_attn.config, '_attn_implementation', None)}"
      )
      print(
          "[DEBUG_RUN_PYTORCH] layer 0 self_attn type:"
          f" {type(model.model.language_model.layers[0].self_attn)}"
      )

  model.load_state_dict(torch.load(os.path.join(tmp_dir, "weights.pt")))
  model = model.to(device)
  model.eval()

  inputs = torch.load(os.path.join(tmp_dir, "inputs.pt"))
  inputs = {k: v.to(device) for k, v in inputs.items()}

  # Use eager model for debugging
  # compiled_model = torch.compile(model, backend="tpu", dynamic=False)

  # Warmup
  with torch.no_grad():
    _ = model(**inputs)

  # Run
  with torch.no_grad():
    out = model(**inputs)
    logits = out.logits.cpu()

  torch.save(logits, os.path.join(tmp_dir, "pytorch_out.pt"))


def prepare_for_torchax(model, inputs):
  import jax
  import torchax

  default_jax_device = jax.devices()[0]
  memo = {}
  torchax.default_env().config.use_dlpack_for_data_conversion = False

  with jax.default_device(default_jax_device):

    def _move(module):
      for name, param in list(module.named_parameters(recurse=False)):
        param_id = id(param)
        if param_id in memo:
          module.register_parameter(name, memo[param_id])
        else:
          param_jax_data = param.data.to("jax")
          new_param = torch.nn.Parameter(param_jax_data, param.requires_grad)
          module.register_parameter(name, new_param)
          memo[param_id] = new_param

      for k in dir(module):
        try:
          v = getattr(module, k)
        except:
          continue
        if isinstance(v, torch.Tensor) and not isinstance(
            v, torch.nn.Parameter
        ):
          setattr(module, k, v.to("jax"))

      for child in module.children():
        _move(child)

    _move(model)

    def _move_tensor(x):
      if isinstance(x, torch.Tensor):
        res = x.to("jax")
        return res
      return x

    inputs_jax = jax.tree_util.tree_map(_move_tensor, inputs)

  return inputs_jax


def custom_sdpa_upcast(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
  import math

  if DEBUG:
    print(
        f"[DEBUG_CUSTOM_SDPA] Called | Q shape: {query.shape}, K shape:"
        f" {key.shape}, V shape: {value.shape} | is_causal: {is_causal} |"
        f" scale: {scale}"
    )
    print(
        f"[DEBUG_CUSTOM_SDPA] Q mean: {query.mean().item():.6f}, K mean:"
        f" {key.mean().item():.6f}, V mean: {value.mean().item():.6f}"
    )
    if attn_mask is not None:
      print(
          "[DEBUG_CUSTOM_SDPA] attn_mask mean:"
          f" {attn_mask.float().mean().item():.6f}"
      )
    else:
      print(f"[DEBUG_CUSTOM_SDPA] attn_mask is None")

  L, S = query.size(-2), key.size(-2)
  scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
  attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
  if is_causal:
    assert attn_mask is None
    temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(
        diagonal=0
    )
    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    attn_bias.to(query.dtype)
  if attn_mask is not None:
    if attn_mask.dtype == torch.bool:
      attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
    else:
      attn_bias += attn_mask
  if enable_gqa:
    key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
    value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

  attn_weight = query @ key.transpose(-2, -1) * scale_factor
  attn_weight += attn_bias

  attn_weight_f32 = attn_weight.to(torch.float32)
  attn_weight_f32 = torch.softmax(attn_weight_f32, dim=-1)
  attn_weight = attn_weight_f32.to(query.dtype)

  if dropout_p > 0:
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
  out = attn_weight @ value
  if DEBUG:
    print(f"[DEBUG_CUSTOM_SDPA] OUT mean: {out.mean().item():.6f}")
  return out


def run_torchax_sub():
  # pylint: disable=g-import-not-at-top
  import numpy as np
  import torchax

  # Enable TorchAX
  torchax.enable_globally()
  torchax.default_env().override_op_definition(
      torch.nn.functional.scaled_dot_product_attention,
      custom_sdpa_upcast,
  )
  torchax.enable_performance_mode()

  tmp_dir = get_tmp_dir()

  module_spec = get_shrunken_module_spec(FLAGS.model_name, "sdpa")
  model = module_spec.module_factory().to(torch.float32)
  print(
      "[DEBUG_RUN_TORCHAX] model config _attn_implementation:"
      f" {getattr(model.config, '_attn_implementation', None)}"
  )
  if hasattr(model, "model") and hasattr(model.model, "language_model"):
    print(
        "[DEBUG_RUN_TORCHAX] model.model config _attn_implementation:"
        f" {getattr(model.model.config, '_attn_implementation', None)}"
    )
    print(
        "[DEBUG_RUN_TORCHAX] model.model.language_model config"
        " _attn_implementation:"
        f" {getattr(model.model.language_model.config, '_attn_implementation', None)}"
    )
    if len(model.model.language_model.layers) > 0:
      print(
          "[DEBUG_RUN_TORCHAX] layer 0 config _attn_implementation:"
          f" {getattr(model.model.language_model.layers[0].self_attn.config, '_attn_implementation', None)}"
      )
      print(
          "[DEBUG_RUN_TORCHAX] layer 0 self_attn type:"
          f" {type(model.model.language_model.layers[0].self_attn)}"
      )

  model.load_state_dict(torch.load(os.path.join(tmp_dir, "weights.pt")))
  model.eval()

  inputs = torch.load(os.path.join(tmp_dir, "inputs.pt"))

  # Prepare model and inputs for TorchAX (moves to JAX)
  inputs = prepare_for_torchax(model, inputs)

  # Warmup eager
  with torch.no_grad():
    _ = model(**inputs)

  # Run eager
  with torch.no_grad():
    out = model(**inputs)
    logits = out.logits
    if hasattr(logits, "_elem") and hasattr(logits._elem, "block_until_ready"):
      logits._elem.block_until_ready()
    elif hasattr(logits, "block_until_ready"):
      logits.block_until_ready()

  # Convert to standard PyTorch CPU tensor
  if hasattr(logits, "to"):
    logits = logits.to("cpu")
  else:
    if hasattr(logits, "jax"):
      logits = torch.from_numpy(np.array(logits.jax()))
    else:
      logits = torch.from_numpy(np.array(logits))

  torch.save(logits, os.path.join(tmp_dir, "torchax_out.pt"))


def run_kernel_pytorch_sub():
  from torch_tpu import _loader

  _loader.load()
  from examples.gemma4 import splash_attention

  tmp_dir = get_tmp_dir()
  inputs = torch.load(os.path.join(tmp_dir, "kernel_inputs.pt"))
  q = inputs["q"].to("tpu")
  k = inputs["k"].to("tpu")
  v = inputs["v"].to("tpu")

  scale = inputs["scale"]
  local_window_size = inputs["local_window_size"]
  enable_gqa = inputs["enable_gqa"]

  out = splash_attention.splash_sdpa(
      q,
      k,
      v,
      scale=scale,
      local_window_size=local_window_size,
      enable_gqa=enable_gqa,
  )

  torch.save(out.cpu(), os.path.join(tmp_dir, "out_kernel_pt.pt"))


def run_kernel_jax_sub():
  import jax
  import jax.numpy as jnp
  import numpy as np
  import math
  from examples.gemma4 import splash_attention

  tmp_dir = get_tmp_dir()
  inputs = torch.load(os.path.join(tmp_dir, "kernel_inputs.pt"))

  import jax.dlpack as jax_dlpack

  q_jax_cpu = jax_dlpack.from_dlpack(inputs["q"])
  k_jax_cpu = jax_dlpack.from_dlpack(inputs["k"])
  v_jax_cpu = jax_dlpack.from_dlpack(inputs["v"])

  tpu_dev = jax.devices("tpu")[0]
  q_jax = jax.device_put(q_jax_cpu, tpu_dev)
  k_jax = jax.device_put(k_jax_cpu, tpu_dev)
  v_jax = jax.device_put(v_jax_cpu, tpu_dev)

  scale = inputs["scale"]
  local_window_size = inputs["local_window_size"]
  enable_gqa = inputs["enable_gqa"]

  batch, n_heads, seq_len, head_dim = q_jax.shape
  n_kv_heads = k_jax.shape[1]

  if n_kv_heads == 1:
    k_jax = jnp.squeeze(k_jax, axis=1)
    v_jax = jnp.squeeze(v_jax, axis=1)

  splash_args = dict(
      seq_len=seq_len,
      n_heads=n_heads,
      n_kv_heads=n_kv_heads,
      is_causal=True,
      local_window_size=local_window_size,
  )
  splash_fn, _ = splash_attention._make_splash_attention_fn(**splash_args)

  default_scale = 1.0 / math.sqrt(head_dim)
  if scale is not None:
    if abs(scale - default_scale) > 1e-6:
      scale_ratio = scale / default_scale
      q_jax = q_jax * scale_ratio

  out_jax, _ = splash_fn(q_jax, k_jax, v_jax)
  out_jax.block_until_ready()

  out_jax_cpu = jax.device_put(out_jax, jax.devices("cpu")[0])
  out_pt = torch.from_dlpack(out_jax_cpu)

  torch.save(out_pt, os.path.join(tmp_dir, "out_kernel_jax.pt"))


def run_full_attn_pytorch_sub():
  from torch_tpu import _loader

  _loader.load()

  tmp_dir = get_tmp_dir()
  inputs = torch.load(os.path.join(tmp_dir, "full_attn_inputs.pt"))
  q = inputs["q"].to("tpu")
  k = inputs["k"].to("tpu")
  v = inputs["v"].to("tpu")

  out = torch.nn.functional.scaled_dot_product_attention(
      q, k, v, is_causal=True
  )

  torch.save(out.cpu(), os.path.join(tmp_dir, "out_full_attn_pt.pt"))


def run_full_attn_jax_sub():
  import jax
  import torchax

  torchax.enable_globally()
  torchax.default_env().override_op_definition(
      torch.nn.functional.scaled_dot_product_attention,
      custom_sdpa_upcast,
  )
  torchax.enable_performance_mode()

  tmp_dir = get_tmp_dir()
  inputs = torch.load(os.path.join(tmp_dir, "full_attn_inputs.pt"))

  import jax.dlpack as jax_dlpack

  q_jax_cpu = jax_dlpack.from_dlpack(inputs["q"])
  k_jax_cpu = jax_dlpack.from_dlpack(inputs["k"])
  v_jax_cpu = jax_dlpack.from_dlpack(inputs["v"])

  tpu_dev = jax.devices("tpu")[0]
  q_jax = jax.device_put(q_jax_cpu, tpu_dev)
  k_jax = jax.device_put(k_jax_cpu, tpu_dev)
  v_jax = jax.device_put(v_jax_cpu, tpu_dev)

  q_pt = torchax.tensor.Tensor(q_jax, torchax.default_env())
  k_pt = torchax.tensor.Tensor(k_jax, torchax.default_env())
  v_pt = torchax.tensor.Tensor(v_jax, torchax.default_env())

  out_pt = torch.nn.functional.scaled_dot_product_attention(
      q_pt, k_pt, v_pt, is_causal=True
  )

  out_jax = out_pt._elem
  out_jax_cpu = jax.device_put(out_jax, jax.devices("cpu")[0])
  out_std_pt = torch.from_dlpack(out_jax_cpu)

  torch.save(out_std_pt, os.path.join(tmp_dir, "out_full_attn_jax.pt"))


def run_rmsnorm_pytorch_sub():
  from torch_tpu import _loader

  _loader.load()

  tmp_dir = get_tmp_dir()
  inputs = torch.load(os.path.join(tmp_dir, "rmsnorm_inputs.pt"))
  x = inputs["x"].to("tpu")
  weight = inputs["weight"].to("tpu")

  norm = Gemma4RMSNorm(dim=x.shape[-1], eps=1e-6).to("tpu")
  norm.weight.data.copy_(weight)

  out = norm(x)
  torch.save(out.cpu(), os.path.join(tmp_dir, "out_rmsnorm_pt.pt"))


def run_rmsnorm_jax_sub():
  import jax
  import numpy as np
  import torchax

  torchax.enable_globally()
  torchax.enable_performance_mode()

  tmp_dir = get_tmp_dir()
  inputs = torch.load(os.path.join(tmp_dir, "rmsnorm_inputs.pt"))

  norm = Gemma4RMSNorm(dim=inputs["x"].shape[-1], eps=1e-6)
  norm.weight.data.copy_(inputs["weight"])

  prepare_for_torchax(norm, None)

  from torchax.ops import mappings

  tpu_dev = jax.devices("tpu")[0]
  x_jax = jax.device_put(mappings.t2j(inputs["x"], use_dlpack=False), tpu_dev)
  x_pt = torchax.tensor.Tensor(x_jax, torchax.default_env())

  out_pt = norm(x_pt)

  out_jax = out_pt._elem
  out_jax_cpu = jax.device_put(out_jax, jax.devices("cpu")[0])
  out_std_pt = torch.from_dlpack(out_jax_cpu)
  torch.save(out_std_pt, os.path.join(tmp_dir, "out_rmsnorm_jax.pt"))


def run_rope_pytorch_sub():
  from torch_tpu import _loader

  _loader.load()

  tmp_dir = get_tmp_dir()
  inputs = torch.load(os.path.join(tmp_dir, "rope_inputs.pt"))
  x = inputs["x"].to("tpu")
  cos = inputs["cos"].to("tpu")
  sin = inputs["sin"].to("tpu")

  out = apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=2)
  torch.save(out.cpu(), os.path.join(tmp_dir, "out_rope_pt.pt"))


def run_rope_jax_sub():
  import jax
  import numpy as np
  import torchax

  torchax.enable_globally()
  torchax.enable_performance_mode()

  tmp_dir = get_tmp_dir()
  inputs = torch.load(os.path.join(tmp_dir, "rope_inputs.pt"))

  from torchax.ops import mappings

  tpu_dev = jax.devices("tpu")[0]
  x_jax = jax.device_put(mappings.t2j(inputs["x"], use_dlpack=False), tpu_dev)
  cos_jax = jax.device_put(
      mappings.t2j(inputs["cos"], use_dlpack=False), tpu_dev
  )
  sin_jax = jax.device_put(
      mappings.t2j(inputs["sin"], use_dlpack=False), tpu_dev
  )

  x_pt = torchax.tensor.Tensor(x_jax, torchax.default_env())
  cos_pt = torchax.tensor.Tensor(cos_jax, torchax.default_env())
  sin_pt = torchax.tensor.Tensor(sin_jax, torchax.default_env())

  out_pt = apply_rotary_pos_emb(x_pt, cos_pt, sin_pt, unsqueeze_dim=2)

  out_jax = out_pt._elem
  out_jax_cpu = jax.device_put(out_jax, jax.devices("cpu")[0])
  out_std_pt = torch.from_dlpack(out_jax_cpu)
  torch.save(out_std_pt, os.path.join(tmp_dir, "out_rope_jax.pt"))


def run_mlp_pytorch_sub():
  from torch_tpu import _loader

  _loader.load()

  tmp_dir = get_tmp_dir()
  inputs = torch.load(os.path.join(tmp_dir, "mlp_inputs.pt"))
  x = inputs["x"].to("tpu")

  module_spec = get_shrunken_module_spec(FLAGS.model_name)
  config = module_spec.config.text_config

  mlp = Gemma4TextMLP(config, layer_idx=0).to(
      device="tpu", dtype=torch.bfloat16
  )

  mlp.gate_proj.weight.data.copy_(inputs["gate_weight"])
  mlp.up_proj.weight.data.copy_(inputs["up_weight"])
  mlp.down_proj.weight.data.copy_(inputs["down_weight"])

  out = mlp(x)
  torch.save(out.cpu(), os.path.join(tmp_dir, "out_mlp_pt.pt"))


def run_mlp_jax_sub():
  import jax
  import torchax

  torchax.enable_globally()
  torchax.enable_performance_mode()

  tmp_dir = get_tmp_dir()
  inputs = torch.load(os.path.join(tmp_dir, "mlp_inputs.pt"))

  module_spec = get_shrunken_module_spec(FLAGS.model_name)
  config = module_spec.config.text_config

  mlp = Gemma4TextMLP(config, layer_idx=0).to(torch.bfloat16)
  mlp.gate_proj.weight.data.copy_(inputs["gate_weight"])
  mlp.up_proj.weight.data.copy_(inputs["up_weight"])
  mlp.down_proj.weight.data.copy_(inputs["down_weight"])

  prepare_for_torchax(mlp, None)

  from torchax.ops import mappings

  tpu_dev = jax.devices("tpu")[0]
  x_jax = jax.device_put(mappings.t2j(inputs["x"], use_dlpack=False), tpu_dev)
  x_pt = torchax.tensor.Tensor(x_jax, torchax.default_env())

  out_pt = mlp(x_pt)

  out_jax = out_pt._elem
  out_jax_cpu = jax.device_put(out_jax, jax.devices("cpu")[0])
  out_std_pt = torch.from_dlpack(out_jax_cpu)
  torch.save(out_std_pt, os.path.join(tmp_dir, "out_mlp_jax.pt"))


def run_decoder_pytorch_sub():
  from torch_tpu import _loader

  _loader.load()
  import torch.nn.functional as F

  F.scaled_dot_product_attention = custom_sdpa_upcast

  tmp_dir = get_tmp_dir()
  layer_idx = FLAGS.layer_idx
  inputs = torch.load(os.path.join(tmp_dir, f"decoder_inputs_{layer_idx}.pt"))

  hidden_states = inputs["hidden_states"].to("tpu")
  per_layer_input = inputs["per_layer_input"].to("tpu")
  cos = inputs["cos"].to("tpu")
  sin = inputs["sin"].to("tpu")
  position_embeddings = (cos, sin)

  module_spec = get_shrunken_module_spec(FLAGS.model_name)
  model = module_spec.module_factory().to(torch.bfloat16)
  model.load_state_dict(torch.load(os.path.join(tmp_dir, "decoder_weights.pt")))
  model = model.to("tpu")
  if hasattr(model.model, "language_model"):
    decoder_layer = model.model.language_model.layers[layer_idx]
  else:
    decoder_layer = model.model.layers[layer_idx]

  from collections import UserDict

  shared_kv_states = UserDict()
  if "shared_kv_states" in inputs:
    for k, (v_k, v_v) in inputs["shared_kv_states"].items():
      shared_kv_states[k] = (v_k.to("tpu"), v_v.to("tpu"))

  decoder_layer.eval()

  attention_mask = inputs.get("attention_mask", None)
  if attention_mask is not None:
    attention_mask = attention_mask.to("tpu")

  with torch.no_grad():
    out = decoder_layer(
        hidden_states=hidden_states,
        per_layer_input=per_layer_input,
        shared_kv_states=shared_kv_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
    )

  torch.save(out.cpu(), os.path.join(tmp_dir, f"out_decoder_pt_{layer_idx}.pt"))


def run_decoder_jax_sub():
  import jax
  import torchax

  torchax.enable_globally()
  torchax.default_env().override_op_definition(
      torch.nn.functional.scaled_dot_product_attention,
      custom_sdpa_upcast,
  )
  torchax.enable_performance_mode()

  tmp_dir = get_tmp_dir()
  layer_idx = FLAGS.layer_idx
  inputs = torch.load(os.path.join(tmp_dir, f"decoder_inputs_{layer_idx}.pt"))

  module_spec = get_shrunken_module_spec(FLAGS.model_name, "sdpa")
  model = module_spec.module_factory().to(torch.bfloat16)
  model.load_state_dict(torch.load(os.path.join(tmp_dir, "decoder_weights.pt")))

  prepare_for_torchax(model, None)

  if hasattr(model.model, "language_model"):
    decoder_layer = model.model.language_model.layers[layer_idx]
  else:
    decoder_layer = model.model.layers[layer_idx]
  decoder_layer.eval()

  from torchax.ops import mappings

  tpu_dev = jax.devices("tpu")[0]

  hidden_states_jax = jax.device_put(
      mappings.t2j(inputs["hidden_states"], use_dlpack=False), tpu_dev
  )
  per_layer_input_jax = jax.device_put(
      mappings.t2j(inputs["per_layer_input"], use_dlpack=False), tpu_dev
  )
  cos_jax = jax.device_put(
      mappings.t2j(inputs["cos"], use_dlpack=False), tpu_dev
  )
  sin_jax = jax.device_put(
      mappings.t2j(inputs["sin"], use_dlpack=False), tpu_dev
  )

  hidden_states_pt = torchax.tensor.Tensor(
      hidden_states_jax, torchax.default_env()
  )
  per_layer_input_pt = torchax.tensor.Tensor(
      per_layer_input_jax, torchax.default_env()
  )
  cos_pt = torchax.tensor.Tensor(cos_jax, torchax.default_env())
  sin_pt = torchax.tensor.Tensor(sin_jax, torchax.default_env())
  position_embeddings = (cos_pt, sin_pt)

  from collections import UserDict

  shared_kv_states = UserDict()
  if "shared_kv_states" in inputs:
    for k, (v_k, v_v) in inputs["shared_kv_states"].items():
      v_k_jax = jax.device_put(mappings.t2j(v_k, use_dlpack=False), tpu_dev)
      v_v_jax = jax.device_put(mappings.t2j(v_v, use_dlpack=False), tpu_dev)
      shared_kv_states[k] = (
          torchax.tensor.Tensor(v_k_jax, torchax.default_env()),
          torchax.tensor.Tensor(v_v_jax, torchax.default_env()),
      )

  attention_mask = inputs.get("attention_mask", None)
  attention_mask_pt = None
  if attention_mask is not None:
    attention_mask_jax = jax.device_put(
        mappings.t2j(attention_mask, use_dlpack=False), tpu_dev
    )
    attention_mask_pt = torchax.tensor.Tensor(
        attention_mask_jax, torchax.default_env()
    )

  with torch.no_grad():
    out_pt = decoder_layer(
        hidden_states=hidden_states_pt,
        per_layer_input=per_layer_input_pt,
        shared_kv_states=shared_kv_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask_pt,
    )

  out_jax = out_pt._elem
  out_jax_cpu = jax.device_put(out_jax, jax.devices("cpu")[0])
  out_std_pt = torch.from_dlpack(out_jax_cpu)
  torch.save(
      out_std_pt, os.path.join(tmp_dir, f"out_decoder_jax_{layer_idx}.pt")
  )


def main(argv):
  if FLAGS.run_mode == "runner":
    absltest.main(argv=argv)
  elif FLAGS.run_mode == "pytorch":
    run_pytorch_sub()
  elif FLAGS.run_mode == "torchax":
    run_torchax_sub()
  elif FLAGS.run_mode == "kernel_pytorch":
    run_kernel_pytorch_sub()
  elif FLAGS.run_mode == "kernel_jax":
    run_kernel_jax_sub()
  elif FLAGS.run_mode == "full_attn_pytorch":
    run_full_attn_pytorch_sub()
  elif FLAGS.run_mode == "full_attn_jax":
    run_full_attn_jax_sub()
  elif FLAGS.run_mode == "rmsnorm_pytorch":
    run_rmsnorm_pytorch_sub()
  elif FLAGS.run_mode == "rmsnorm_jax":
    run_rmsnorm_jax_sub()
  elif FLAGS.run_mode == "rope_pytorch":
    run_rope_pytorch_sub()
  elif FLAGS.run_mode == "rope_jax":
    run_rope_jax_sub()
  elif FLAGS.run_mode == "mlp_pytorch":
    run_mlp_pytorch_sub()
  elif FLAGS.run_mode == "mlp_jax":
    run_mlp_jax_sub()
  elif FLAGS.run_mode == "decoder_pytorch":
    run_decoder_pytorch_sub()
  elif FLAGS.run_mode == "decoder_jax":
    run_decoder_jax_sub()
  elif FLAGS.run_mode == "splash_pytorch":
    run_splash_pytorch_sub()


def run_splash_pytorch_sub():
  from torch_tpu import _loader

  _loader.load()
  from examples.gemma4 import splash_attention

  tmp_dir = get_tmp_dir()
  inputs = torch.load(os.path.join(tmp_dir, "splash_vs_ref_inputs.pt"))
  q = inputs["q"].to("tpu")
  k = inputs["k"].to("tpu")
  v = inputs["v"].to("tpu")
  window_size = inputs["window_size"]

  head_dim = q.size(-1)
  scaling = head_dim**-0.5

  seq_len = q.size(-2)
  block_size = min(512, seq_len)
  block_size = max(128, block_size)

  out = splash_attention.splash_sdpa(
      q,
      k,
      v,
      scale=scaling,
      local_window_size=window_size - 1,
      enable_gqa=True,
      block_q=block_size,
      block_kv=block_size,
  )

  torch.save(out.cpu(), os.path.join(tmp_dir, "out_splash_pt.pt"))


if __name__ == "__main__":
  app.run(main)
