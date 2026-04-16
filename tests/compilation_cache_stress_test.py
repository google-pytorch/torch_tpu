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

"""Stress test for compilation cache eviction."""

import concurrent.futures
import time

from absl.testing import absltest
import torch
from torch_tpu._internal.compile import tpu_torch_compile


class CompilationCacheStressTest(absltest.TestCase):

  def test_eviction_does_not_affect_inflight_compilation(self):
    """Eviction should not affect in-flight compilations."""

    # Use many threads to increase the chance of triggering the race.
    num_compile_threads = 20
    num_iterations = 50

    def trigger_compilation(i):
      # Create a unique MLIR module to ensure a cache miss and trigger
      # compilation. We use a simple addition with different constants
      # to change the hash.
      mlir_text = f"""
module {{
  func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {{
    %0 = stablehlo.constant dense<{i}.0> : tensor<10xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<10xf32>
    return %1 : tensor<10xf32>
  }}
}}
"""
      # Parse the MLIR text into an MLIR module.
      mlir_module = tpu_torch_compile.parse_mlir_text(mlir_text)
      # Trigger background compilation.
      tpu_torch_compile.compile_mlir(mlir_module, fast_compile=False)

    def evict_cache():
      for _ in range(num_iterations * 2):
        # Repeatedly clear the cache while compilations are in flight.
        torch.tpu._clear_cache()
        time.sleep(0.01)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_compile_threads + 1  # + 1 for eviction thread.
    ) as executor:
      # Start eviction thread.
      eviction_future = executor.submit(evict_cache)

      # Start many compilation threads.
      compilation_futures = []
      for i in range(num_iterations):
        for _ in range(num_compile_threads):
          compilation_futures.append(executor.submit(trigger_compilation, i))
        time.sleep(0.05)

      # Wait for all compilations to complete successfully.
      for future in compilation_futures:
        future.result()

      eviction_future.result()


if __name__ == '__main__':
  absltest.main()
