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
# pylint: skip-file

import pytest


def test_factory_demo():
  import torch

  import tpu_utils

  device = tpu_utils.safe_init()

  # This is fused — no physical TPU compute is used yet
  t1 = torch.ones((10, 10), device="tpu", dtype=torch.bfloat16)
  print(f"t1 shape: {t1.shape}, dtype: {t1.dtype}")
  print("  (Fused — no TPU compute has happened yet)")

  # This is also fused — still no hardware execution
  t2 = torch.arange(100, device="tpu", dtype=torch.bfloat16)
  t3 = t1 + t2.reshape(10, 10)
  print(f"\nt3 shape: {t3.shape}")
  print("  (Still fused — the operations are waiting to be optimized)")

  # Transferring to CPU triggers the execution of the fused operations
  result = t3.cpu()
  print(f"\nresult shape: {result.shape}")
  print(
      "  (Executed — fused operations were optimized and executed, data copied"
      " to CPU)"
  )
  return (torch,)


def test_item_demo():
  import torch

  print("\n--- .item() Execution Demo ---")

  # Create two scalar tensors
  v1 = torch.tensor(5.0, device="tpu", dtype=torch.bfloat16)
  v2 = torch.tensor(3.0, device="tpu", dtype=torch.bfloat16)
  print("  (Scalars created — fused)")

  # Perform a comparison
  cond = v1 > v2
  print("  (Comparison recorded — fused)")

  # Using .item() triggers execution to get the python boolean
  is_greater = cond.item()
  print(f"  (Executed via .item() — Result: {is_greater})")

  # Using the tensor in an `if` statement implicitly triggers execution
  print("  (Evaluating `if cond:`...)")
  if cond:
    print("  (Executed implicitly via control flow)")
  return


def test_modes_demo():
  import torch
  from torch_tpu._internal import execution_mode as em
  from torch_tpu._internal.sync import sync

  def check_fusion(mode_name, mode):
    print(f"--- Mode: {mode_name} ---")
    with em.eager_mode(mode):
      x = torch.ones((2, 2), device="tpu")
      y = x + 1
      z = y * 2

      # computation_mlir returns the StableHLO graph for the given tensor.
      # If fused, it will contain 'add' and 'multiply' operations.
      mlir = sync.computation_mlir(z)
      has_add = "stablehlo.add" in mlir
      has_mul = "stablehlo.multiply" in mlir

      print(f"  Graph contains fused 'add': {has_add}")
      print(f"  Graph contains fused 'multiply': {has_mul}")
      if has_add and has_mul:
        print("  Result: Operations are FUSED into a single graph.")
      else:
        print("  Result: Operations were EXECUTED immediately (not fused).")
    print()

  check_fusion("Fused Eager (OPTIMIZED)", em.EagerMode.OPTIMIZED)
  check_fusion("Strict Eager (DEFER_NEVER)", em.EagerMode.DEFER_NEVER)
  return
