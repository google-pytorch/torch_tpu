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

"""Distributed Llama model benchmark definition.

Contains multi-device FairScale tensor-parallel Llama benchmarks (such as
`meta_llama_8b_forward`) that are discovered and executed by
`distributed_runner.py` on multi-chip TPU (`v7_2x2x1`, `v7_2x2x2`, etc.) and
multi-GPU (`b200_4`, `b200_8`) slices. Single-device HuggingFace Llama
benchmarks live in `models/llama.py` and are run by `base_test.py`.
"""

import math

from fairscale.nn.model_parallel import initialize as fairscale_init
import llama_models.llama3.model as llama_model
import torch
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab import torch_device_ops


def _init_model_weights(model: torch.nn.Module) -> None:
  """Initializes model weights to small random values."""
  with torch.no_grad():
    if hasattr(model, "weight") and model.weight is not None:
      tensor = model.weight.data
      tensor.normal_(std=0.01)

    if hasattr(model, "bias") and model.bias is not None:
      tensor = model.bias.data
      tensor.fill_(0.0)


def _load_meta_llama(
    ctx: context_lib.Context,
    model_name: str,
) -> tuple[torch.nn.Module, tuple[torch.Tensor, int]]:
  """Loads the Meta Llama model with FairScale Tensor Parallelism.

  This function initializes a Meta Llama model with random weights and prepares
  example inputs suitable for benchmarking. Model parallel initialization is
  handled internally based on the distributed world size.

  Args:
    ctx: Benchmark execution context containing device and run scope options.
    model_name: The Meta Llama model variant (e.g. 'Llama-3.2-8B').

  Returns:
    A tuple of (model, example_inputs) ready for execution.
  """
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
  else:
    batch_size = 1
    seq_len = 2048

  if model_name == "Llama-3.2-8B":
    args = llama_model.ModelArgs(
        dim=4096,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        n_heads=32,
        n_kv_heads=8,
        n_layers=32,
        norm_eps=1e-05,
        rope_theta=500000.0,
        use_scaled_rope=True,
        vocab_size=128256,
        max_seq_len=2048,
        max_batch_size=batch_size,
    )
  elif model_name == "Llama-3.2-70B":
    args = llama_model.ModelArgs(
        dim=8192,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        n_heads=64,
        n_kv_heads=8,
        n_layers=80,
        norm_eps=1e-05,
        rope_theta=500000.0,
        use_scaled_rope=True,
        vocab_size=128256,
        max_seq_len=2048,
        max_batch_size=batch_size,
    )
  else:
    raise ValueError(f"Unknown model name: {model_name}")

  if not torch.distributed.is_initialized():
    raise target_lib.UnsupportedBenchmark(
        "Meta Llama benchmark requires an initialized torch.distributed"
        " environment for FairScale model parallelism."
    )

  world_size = torch.distributed.get_world_size()
  n_kv_heads = args.n_kv_heads or args.n_heads
  mp_size = math.gcd(world_size, n_kv_heads)

  # Ensure model parallel is initialized
  if not fairscale_init.model_parallel_is_initialized():
    fairscale_init.initialize_model_parallel(mp_size)

  dtype = torch_device_ops.get_torch_dtype(ctx.dtype)
  prev_dtype = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  try:
    device = torch.device(ctx.device_kind.value)
    # Initialize on CPU first and then transfer to `device`. Upstream PyPI
    # `fairscale==0.4.13` allocates `self.weight` via `torch.Tensor(...)` (which
    # ignores `with torch.device(device):` and stays on CPU) while allocating
    # `master_weight` via `torch.empty(...)` (which DeviceContext places on
    # `device`), causing `torch.cat(my_weight_list, out=self.weight)` to fail
    # with a cross-device `out=` tensor if constructed inside `with device:`.
    with torch.inference_mode():
      model = llama_model.Transformer(args)
      model = model.to(device=device)
      model.freqs_cis = model.freqs_cis.to(device=device)
      for submodule in model.modules():
        if isinstance(submodule, llama_model.Attention):
          submodule.cache_k = submodule.cache_k.to(device=device, dtype=dtype)
          submodule.cache_v = submodule.cache_v.to(device=device, dtype=dtype)
      model.apply(_init_model_weights)

    input_ids = torch.randint(
        0, args.vocab_size, (batch_size, seq_len), device=device
    )
    # The model expects (tokens, start_pos)
    example_inputs = (input_ids, 0)
    return model, example_inputs
  finally:
    torch.set_default_dtype(prev_dtype)


@registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
def meta_llama_8b_forward(ctx: context_lib.Context):
  """Benchmark factory for Meta Llama 3.2 8B forward across single-host and multi-host slices."""
  supported = [
      target_lib.Platform.V7_2X2X1,
      target_lib.Platform.V7_2X2X2,
      target_lib.Platform.V7_2X2X4,
      target_lib.Platform.V7_2X4X4,
      target_lib.Platform.B200_4,
      target_lib.Platform.B200_8,
  ]
  if ctx.target.platform not in supported:
    raise target_lib.UnsupportedBenchmark(
        f"Llama 8B is not supported on platform {ctx.target.platform}"
    )

  model, example_inputs = _load_meta_llama(ctx, "Llama-3.2-8B")
  return model, example_inputs, {}
