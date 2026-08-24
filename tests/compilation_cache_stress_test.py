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
import sys
import threading
import time

from absl.testing import absltest
import torch
from torch_tpu.csrc.internal.compile import tpu_torch_compile
from tests import seed_test_utils


class CompilationCacheStressTest(seed_test_utils.RepeatableTest):

  def test_compile_mlir_releases_gil(self):
    """Python threads should run while PJRT compilation is in progress."""
    # Make the module unique to avoid a persistent compilation-cache hit. The
    # chain also keeps compilation in flight long enough for the waiting thread
    # to be scheduled after the native binding releases the GIL.
    constant = time.time_ns() % 1_000_000
    operations = [
        f"  %c = stablehlo.constant dense<{constant}.0> : tensor<256x256xf32>",
        "  %0 = stablehlo.add %arg0, %c : tensor<256x256xf32>",
    ]
    for index in range(1, 200):
      operations.append(
          f"  %{index} = stablehlo.add %{index - 1}, %c : tensor<256x256xf32>"
      )
    mlir_text = "\n".join([
        "module {",
        "func.func @main(%arg0: tensor<256x256xf32>) -> tensor<256x256xf32> {",
        *operations,
        "  return %199 : tensor<256x256xf32>",
        "}",
        "}",
    ])
    mlir_module = tpu_torch_compile.parse_mlir_text(mlir_text)

    compile_started = threading.Event()
    compile_finished = threading.Event()
    heartbeat_ran_during_compile = threading.Event()

    def heartbeat():
      compile_started.wait()
      if not compile_finished.is_set():
        heartbeat_ran_during_compile.set()

    heartbeat_thread = threading.Thread(target=heartbeat)
    heartbeat_thread.start()

    # Prevent the interpreter from scheduling heartbeat between Event.set()
    # and the native call. It must be scheduled by compile_mlir releasing the
    # GIL, not by normal bytecode thread switching.
    previous_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(60.0)
    try:
      compile_started.set()
      tpu_torch_compile.compile_mlir(mlir_module, fast_compile=False)
    finally:
      compile_finished.set()
      sys.setswitchinterval(previous_switch_interval)
      heartbeat_thread.join()

    self.assertTrue(
        heartbeat_ran_during_compile.is_set(),
        "compile_mlir held the Python GIL for the entire PJRT compilation",
    )

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
