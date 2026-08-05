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

import gc
import queue

from absl.testing import absltest
import torch
from torch_tpu._internal import compile as compile_lib
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils


class LargeCheckpointedModel(torch.nn.Module):
  """Larger model to ensure activation memory dominates.

  This model consists of several layers, allowing for a significant amount of
  intermediate activations to be generated, which is necessary to observe
  the memory savings from activation checkpointing.
  """

  def __init__(self):
    super().__init__()
    self.layers = torch.nn.ModuleList([
        torch.nn.Sequential(
            torch.nn.Linear(1024, 4096),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 1024),
        )
        for _ in range(4)
    ])

  def forward(self, x, use_ac):
    for layer in self.layers:
      if use_ac:
        x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
      else:
        x = layer(x)
    return x


def run_and_measure(use_ac):
  device = torch.device("tpu")
  tpu_backend = compile_lib.TpuBackend(debug=True)
  model = LargeCheckpointedModel().to(device)
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, capturable=True)
  criterion = torch.nn.MSELoss()
  compiled = torch.compile(model, backend=tpu_backend)

  input_tensor = torch.randn(32, 512, 1024, device=device)
  target = torch.randn(32, 512, 1024, device=device)

  # Measure
  optimizer.zero_grad()
  outputs = compiled(input_tensor, use_ac)
  loss = criterion(outputs, target)
  loss.backward()
  optimizer.step()

  torch.accelerator.synchronize()
  peak_mem = torch.accelerator.max_memory_allocated()

  del compiled, model, optimizer, criterion, input_tensor, target, outputs, loss
  gc.collect()

  return peak_mem


def _run_and_measure_worker(q: queue.Queue[int | Exception], use_ac: bool):
  try:
    peak = run_and_measure(use_ac)
    q.put(peak)
  except Exception as e:  # pylint: disable=broad-exception-caught
    q.put(e)


class ActivationCheckpointingMemoryTest(
    seed_test_utils.MultiProcessRepeatableTest
):

  def _run_sub_test(self, use_ac: bool) -> int:
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_run_and_measure_worker, args=(q, use_ac))
    p.start()
    p.join()
    if p.exitcode != 0:
      self.fail(f"Worker process failed with exitcode {p.exitcode}")
    result = q.get()
    if isinstance(result, Exception):
      raise result
    return result

  def test_activation_checkpointing_memory_savings(self):
    mem_no_ac = self._run_sub_test(use_ac=False)
    mem_ac = self._run_sub_test(use_ac=True)

    print(f"Peak Memory - No AC: {mem_no_ac / 1e6:.2f} MB")
    print(f"Peak Memory - AC: {mem_ac / 1e6:.2f} MB")

    self.assertLess(
        mem_ac,
        mem_no_ac * 0.9,
        f"AC peak memory ({mem_ac}) is not significantly less than No AC peak"
        f" memory ({mem_no_ac})",
    )


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
