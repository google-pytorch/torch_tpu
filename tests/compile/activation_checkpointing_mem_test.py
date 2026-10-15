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

import contextlib
import gc
import queue
from unittest import mock

from absl.testing import absltest
import torch
from torch_tpu._internal.compile.fx_passes import mark_activation_checkpoints
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


def _train_step(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    use_ac: bool,
) -> torch.Tensor:
  out = model(x, use_ac)
  loss = criterion(out, y)
  loss.backward()
  return loss.detach()


def run_and_measure(
    use_ac: bool, fuse_step: bool = False, use_barrier: bool = True
) -> int:
  device = torch.device("tpu")
  model = LargeCheckpointedModel().to(device)
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, capturable=True)
  criterion = torch.nn.MSELoss()

  input_tensor = torch.randn(32, 512, 1024, device=device)
  target = torch.randn(32, 512, 1024, device=device)

  optimizer.zero_grad()
  outputs = None
  if fuse_step:
    ctx = contextlib.nullcontext()
    if not use_barrier:
      ctx = mock.patch.object(
          mark_activation_checkpoints, "apply", side_effect=lambda gm: None
      )
    with ctx:
      compiled = torch.compile(
          _train_step, backend="tpu", options={"serializable": False}
      )
      loss = compiled(model, criterion, input_tensor, target, use_ac)
  else:
    compiled = torch.compile(
        model,
        backend="tpu",
        options={"serializable": False},
    )
    outputs = compiled(input_tensor, use_ac)
    loss = criterion(outputs, target)
    loss.backward()
  optimizer.step()

  torch.accelerator.synchronize()
  peak_mem = torch.accelerator.max_memory_allocated()

  del compiled, model, optimizer, criterion, input_tensor, target, outputs, loss
  gc.collect()

  return peak_mem


def _run_and_measure_worker(
    q: queue.Queue[int | Exception],
    use_ac: bool,
    fuse_step: bool,
    use_barrier: bool,
):
  try:
    peak = run_and_measure(use_ac, fuse_step=fuse_step, use_barrier=use_barrier)
    q.put(peak)
  except Exception as e:  # pylint: disable=broad-exception-caught
    q.put(e)


class ActivationCheckpointingMemoryTest(
    seed_test_utils.MultiProcessRepeatableTest
):

  def _run_sub_test(
      self, use_ac: bool, fuse_step: bool = False, use_barrier: bool = True
  ) -> int:
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_run_and_measure_worker,
        args=(q, use_ac, fuse_step, use_barrier),
    )
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

    print(f"Peak Memory (Split) - No AC: {mem_no_ac / 1e6:.2f} MB")
    print(f"Peak Memory (Split) - AC: {mem_ac / 1e6:.2f} MB")

    # Empirical threshold: on TPU, split compilation (`torch.compile(model)`)
    # measures ~3288.60 MB without AC vs ~2906.88 MB with AC (~0.884x, -11.6%)
    # because AOTAutograd omits saving the 4 intermediate [32, 512, 4096] GELU
    # activations across the Python boundary between the forward and backward
    # executables, while `outputs` and the remaining saved tensors stay pinned
    # in Python throughout the backward executable's execution.
    self.assertLess(
        mem_ac,
        mem_no_ac * 0.9,
        f"AC peak memory ({mem_ac}) is not significantly less than No AC peak"
        f" memory ({mem_no_ac})",
    )

  def test_fused_step_activation_checkpointing_memory_savings(self):
    mem_no_ac = self._run_sub_test(use_ac=False, fuse_step=True)
    mem_ac_no_barrier = self._run_sub_test(
        use_ac=True, fuse_step=True, use_barrier=False
    )
    mem_ac_barrier = self._run_sub_test(
        use_ac=True, fuse_step=True, use_barrier=True
    )

    print(f"Peak Memory (Fused) - No AC: {mem_no_ac / 1e6:.2f} MB")
    print(
        "Peak Memory (Fused) - AC without barriers:"
        f" {mem_ac_no_barrier / 1e6:.2f} MB"
    )
    print(
        f"Peak Memory (Fused) - AC with barriers: {mem_ac_barrier / 1e6:.2f} MB"
    )

    # Empirical thresholds from TPU measurements on `LargeCheckpointedModel`
    # (kept tight to catch single-layer CSE regressions; widen if compiler
    # scheduling changes across TPU generations):
    # - No AC (`use_ac=False`): ~2269.72 MB
    # - AC without barriers (`use_barrier=False`): ~2269.72 MB (1.000x, 0.0%
    #   savings due to MLIR/XLA CSE keeping forward activations live).
    # - AC with barriers (`use_barrier=True`): ~1533.30 MB (0.676x, -32.4%
    #   savings, freeing ~736 MB of intermediate activations).
    self.assertGreaterEqual(
        mem_ac_no_barrier,
        mem_no_ac * 0.99,
        f"Expected un-barriered fused AC peak memory ({mem_ac_no_barrier}) to"
        f" remain near No AC peak memory ({mem_no_ac}) due to CSE",
    )
    self.assertLess(
        mem_ac_barrier,
        mem_no_ac * 0.75,
        f"Fused AC peak memory ({mem_ac_barrier}) is not at least 25% less"
        f" than No AC peak memory ({mem_no_ac})",
    )


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
