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

"""Reproduction script for PyTorch subclass tracing bug on CPU."""

# pylint: disable=g-import-not-at-top
# pylint: disable=protected-access

import logging  # PYTHON_LOGGING_OK=Configure PyTorch standard logging
import os

# Set environment variables before importing torch
os.environ["TORCH_LOGS"] = "+recompiles,+dynamic,aot_graphs"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.compilers import nop
from torch_tpu._internal.compile.dynamic.state_staged_tensor import (
    StateStagedTensor,
)
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer

logging.basicConfig(level=logging.INFO)


class DummyModel(torch.nn.Module):

  def forward(self, x, past_key_values, attention_mask):
    key_states, value_states = past_key_values.update(x, x, 0)
    # attn_weights = (Q @ K^T) * mask
    # Simplify: assume dot product + mask
    attn_weights = torch.matmul(x, key_states.transpose(-1, -2))
    attn_weights = attn_weights * attention_mask.unsqueeze(1).unsqueeze(2)
    out = torch.matmul(attn_weights, value_states)
    return out, past_key_values


def main():

  class CPUWrappedExecutable:

    def __init__(self, graph_module):
      self.graph_module = graph_module

    def __call__(self, *args):
      arg_info = [
          t.shape if isinstance(t, torch.Tensor) else type(t) for t in args
      ]
      print(f"[CPUWrapper] Executing flat graph with inputs: {arg_info}")
      outputs = self.graph_module(*args)
      return outputs

  def custom_backend(graph_module, example_inputs):
    arg_info = [
        t.shape if isinstance(t, torch.Tensor) else type(t)
        for t in example_inputs
    ]
    print(f"[custom_backend] Called with inputs: {arg_info}")

    def fw_compiler(flat_graph_module, flat_example_inputs):
      flat_arg_info = [
          t.shape if isinstance(t, torch.Tensor) else type(t)
          for t in flat_example_inputs
      ]
      print(
          "[custom_backend] flat fw_compiler called with inputs:"
          f" {flat_arg_info}"
      )
      print("[custom_backend] flat graph code:")
      print(flat_graph_module.code)
      return CPUWrappedExecutable(flat_graph_module)

    return aot_autograd(
        fw_compiler=fw_compiler,
        bw_compiler=nop,
        keep_inference_input_mutations=False,
    )(graph_module, example_inputs)

  model = DummyModel()
  compiled_model = torch.compile(model, backend=custom_backend)

  cache = DynamicCache()
  # Pre-populate cache with empty StateStagedTensor subclasses to force
  # subclass tracing in Step 1.
  layer = DynamicLayer()
  empty_base = torch.zeros(1, 1, 1, 4, 1)
  empty_logical = torch.zeros(1, 1, 0, 1)
  layer.keys = StateStagedTensor(empty_logical, empty_base, 0, 4)
  layer.values = StateStagedTensor(empty_logical, empty_base, 0, 4)
  layer.is_initialized = True
  cache.layers.append(layer)

  # Step 1: Prefill (len 2)
  print("\n--- Step 1 (Prefill) ---")
  x1 = torch.ones(1, 1, 2, 1)
  mask1 = torch.ones(1, 2)
  torch._dynamo.mark_dynamic(x1, 2, min=1, max=3)
  torch._dynamo.mark_dynamic(mask1, 1, min=1, max=3)
  out1, cache = compiled_model(x1, cache, mask1)
  print(f"out1 shape: {out1.shape}")

  print("Cache after step 1 keys type:", type(cache.layers[0].keys))
  print("Cache after step 1 keys shape:", cache.layers[0].keys.shape)
  print(
      "Cache after step 1 keys base shape:",
      cache.layers[0].keys._base_tensor.shape,
  )

  # Step 2: Decode 1 (len 1)
  print("\n--- Step 2 (Decode 1) ---")
  x2 = torch.ones(1, 1, 1, 1)
  mask2 = torch.ones(1, 3)
  keys2 = cache.layers[0].keys
  values2 = cache.layers[0].values
  torch._dynamo.mark_dynamic(keys2, 2, min=1, max=3)
  torch._dynamo.mark_dynamic(values2, 2, min=1, max=3)
  torch._dynamo.mark_dynamic(mask2, 1, min=1, max=4)

  print(
      "keys2 _dynamo_dynamic_indices:",
      getattr(keys2, "_dynamo_dynamic_indices", None),
  )
  print(
      "keys2._base_tensor _dynamo_dynamic_indices:",
      getattr(keys2._base_tensor, "_dynamo_dynamic_indices", None),
  )
  print(
      "keys2 _dynamo_hint_overrides:",
      getattr(keys2, "_dynamo_hint_overrides", None),
  )
  print(
      "keys2._base_tensor _dynamo_hint_overrides:",
      getattr(keys2._base_tensor, "_dynamo_hint_overrides", None),
  )

  out2, cache = compiled_model(x2, cache, mask2)
  print(f"out2 shape: {out2.shape}")
  print("Cache after step 2 keys type:", type(cache.layers[0].keys))
  print("Cache after step 2 keys shape:", cache.layers[0].keys.shape)

  # Step 3: Decode 2 (len 1)
  print("\n--- Step 3 (Decode 2) ---")
  x3 = torch.ones(1, 1, 1, 1)
  mask3 = torch.ones(1, 4)
  keys3 = cache.layers[0].keys
  values3 = cache.layers[0].values
  torch._dynamo.mark_dynamic(keys3, 2, min=1, max=3)
  torch._dynamo.mark_dynamic(values3, 2, min=1, max=3)
  torch._dynamo.mark_dynamic(mask3, 1, min=1, max=4)

  out3, cache = compiled_model(x3, cache, mask3)
  print(f"out3 shape: {out3.shape}")
  print("Cache after step 3 keys shape:", cache.layers[0].keys.shape)

  # Step 4: Decode 3 (len 1)
  print("\n--- Step 4 (Decode 3) ---")
  x4 = torch.ones(1, 1, 1, 1)
  mask4 = torch.ones(1, 5)
  keys4 = cache.layers[0].keys
  values4 = cache.layers[0].values
  torch._dynamo.mark_dynamic(keys4, 2, min=1, max=4)
  torch._dynamo.mark_dynamic(values4, 2, min=1, max=4)
  torch._dynamo.mark_dynamic(mask4, 1, min=1, max=8)

  out4, cache = compiled_model(x4, cache, mask4)
  print(f"out4 shape: {out4.shape}")


if __name__ == "__main__":
  main()
