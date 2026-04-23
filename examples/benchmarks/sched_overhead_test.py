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

"""Micro benchmarks for measuring TorchTPU scheduling overheads."""

from collections.abc import Sequence
import sys
import time

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch._inductor.config as inductor_config
from torch_tpu._internal import execution_mode
from torch_tpu._internal import sync
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.utils import log_utils

from torch_tpu._internal.shims.xprof import traceme
from torch_tpu._internal.shims.xprof import xprof_session


log_utils.log_to_stderr()


_NUM_WARMUP_STEPS = flags.DEFINE_integer(
    "num_warmup_steps", 5, "Numer of warm-up steps."
)

_NUM_STEPS = flags.DEFINE_integer(
    "num_steps", 1000, "Numer of post-warm up steps."
)

_DEVICE = flags.DEFINE_enum(
    "device",
    None,
    ["tpu", "cuda", "xla_cuda"],
    required=True,
    help="Accelerator to test.",
)


def get_torch_device() -> torch.device:
  return torch.device(_DEVICE.value)


def sync_device(
    tensors_to_sync: torch.Tensor | Sequence[torch.Tensor], wait: bool = True
) -> None:
  if _DEVICE.value == "tpu" or _DEVICE.value == "xla_cuda":
    # Wait for the compilation and execution of model output to complete.
    if isinstance(tensors_to_sync, torch.Tensor):
      sync.synchronize(tensors_to_sync, wait=wait)
    else:
      for t in tensors_to_sync:
        sync.synchronize(t, wait=wait)
  elif _DEVICE.value == "cuda":
    if wait:
      torch.cuda.synchronize()


class SchedOverheadTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    cls.logs = []

    if _DEVICE.value in ("cuda", "xla_cuda") and not torch.cuda.is_available():
      print(
          f"WARNING: --device={_DEVICE.value} requires compiling the target"
          " with --config=cuda on the blaze command line. Please rerun the"
          " target with that flag.",
          file=sys.stderr,
      )
      # We return success here to avoid breaking
      sys.exit(0)

    # This abstest flag will always be set to an int.
    seed = absltest.FLAGS.test_random_seed
    if seed is None or not isinstance(seed, int):
      raise ValueError("absltest.FLAGS.test_random_seed not an int: %s" % seed)

    # TODO(gunhyun): Figure out why inductor multiprocessing library is causing
    # issues with GPU.
    if _DEVICE.value == "cuda":
      inductor_config.compile_threads = 1
    torch.manual_seed(seed)
    logging.info("Using absltest.FLAGS.test_random_seed: %d", seed)

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    for log in cls.logs:
      logging.info(log)

  @parameterized.parameters((1), (256), (8192))
  def test_matmul_with_h2d(self, sz):
    device = get_torch_device()
    x = torch.randn([sz, sz], device=device)
    w = torch.randn([sz, sz], device=device)
    for _ in range(_NUM_WARMUP_STEPS.value):
      y = x @ w
      unused_y_cpu = y.to("cpu")  # pylint: disable=unused-variable

    session = xprof_session.XprofSession()
    session.start_session(host_trace_level=3, enable_python_tracer=True)
    cpu_total_time = 0
    loop_start_time = time.time()
    for _ in range(_NUM_STEPS.value):
      with traceme.TraceMe("Eval"):
        cpu_start_time = time.time()
        y = x @ w
        cpu_total_time = time.time() - cpu_start_time
        unused_y_cpu = y.to("cpu")  # pylint: disable=unused-variable
    time_per_step = (time.time() - loop_start_time) / _NUM_STEPS.value
    xprof_url = session.end_session_and_get_url()

    self.logs.append(
        f"H2D: Run for sz={sz} took {time_per_step * 1000} ms per step, tracing"
        f" time: {cpu_total_time * 1000} ms, XProf URL: {xprof_url}"
    )

  @parameterized.parameters((1), (256), (8192))
  def test_matmul_no_h2d(self, sz):
    device = get_torch_device()
    x = torch.randn([sz, sz], device=device)
    w = torch.randn([sz, sz], device=device)
    for _ in range(_NUM_WARMUP_STEPS.value):
      y = x @ w
      sync_device(y, wait=False)
    sync_device(y, wait=True)

    session = xprof_session.XprofSession()
    session.start_session(host_trace_level=3, enable_python_tracer=True)
    cpu_total_time = 0
    loop_start_time = time.time()
    for _ in range(_NUM_STEPS.value):
      with traceme.TraceMe("Eval"):
        cpu_start_time = time.time()
        y = x @ w
        sync_device(y, wait=False)
        cpu_total_time = time.time() - cpu_start_time
    sync_device(y, wait=True)
    time_per_step = (time.time() - loop_start_time) / _NUM_STEPS.value
    xprof_url = session.end_session_and_get_url()

    self.logs.append(
        f"No H2D: Run for sz={sz} took {time_per_step * 1000} ms per step,"
        f" tracing time: {cpu_total_time * 1000} ms, XProf URL: {xprof_url}"
    )

  @parameterized.parameters((1), (16), (1024))
  def test_execute_queueing_overhead_benchmark(self, num_outputs):
    device = get_torch_device()
    with execution_mode.eager_mode(execution_mode.EagerMode.INTERNAL_DEFER_ALL):
      x = torch.ones(10, device="cpu").to(device=device)
      y = torch.ones(10, device="cpu").to(device=device)
      z = x + y

    result_tensors = [z] * num_outputs
    argument_tensors = [x, y]

    mlir = tpu_torch_compile.build_mlir(result_tensors, argument_tensors)
    executable = tpu_torch_compile.compile_mlir(mlir)

    # Warmup
    for _ in range(_NUM_WARMUP_STEPS.value):
      _ = tpu_torch_compile.execute(executable, [x, y])

    loop_start_time = time.time()
    for _ in range(_NUM_STEPS.value):
      _ = tpu_torch_compile.execute(executable, [x, y])
    time_per_step = (time.time() - loop_start_time) / _NUM_STEPS.value

    avg_time_us = time_per_step * 1e6
    self.logs.append(
        f"PyExecuteCompiledModel overhead for {num_outputs} outputs:"
        f" {avg_time_us:.2f} us"
    )


if __name__ == "__main__":
  absltest.main()
