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

"""Benchmark tests for TorchTPU compile mode."""

import concurrent.futures
import functools
import json
import os
import time

from absl import logging
from absl.testing import absltest
import torch
from torch_tpu._internal.utils import device_utils
from torch_tpu._internal.utils import log_utils
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import mlcompass_utils
from examples.benchmarks.e2e import model_utils
from examples.benchmarks.e2e import performance_utils
from examples.benchmarks.e2e import test_utils

from torch_tpu._internal.shims.xprof import traceme

log_utils.log_to_stderr()


@functools.cache
def _get_tpu_duty_cycle(session_id: str) -> float:
  xprof = performance_utils.get_xprof_client()
  assert xprof is not None, "Xprof client should not be None."
  result = xprof.get_profile_data(
      "overview_page.json", session_id=session_id, format="json"
  )
  content = (
      result[1] if (isinstance(result, tuple) and len(result) > 1) else result
  )
  if not content:
    raise RuntimeError(
        f"Failed to retrieve overview_page.json data for session_id: {session_id}"
    )
  overview_data = json.loads(content.decode("utf-8"))
  for section in overview_data:
    p_dict = section.get("p", {})
    if "device_duty_cycle_percent" in p_dict:
      return float(str(p_dict["device_duty_cycle_percent"]).rstrip("%"))
  raise RuntimeError(
      "device_duty_cycle_percent not found in overview sections."
  )


class CompileBenchmarkTest(test_utils.BenchmarkTest):
  logs = []

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.logs = []
    torch.manual_seed(42)

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    for log in sorted(cls.logs):
      logging.info(log)

  def _get_qwen3_1_7b(self, device):
    model_and_input = model_utils.get_huggingface_llm_model(
        "Qwen/Qwen3-1.7B",
        device=device,
        weights_dtype=torch.bfloat16,
        is_training=False,
        use_torch_compile=False,
        sequence_length=256,
        batch_size=1,
        dist_strat=model_utils.DistStrat.NONE,
    )
    model = model_and_input.model

    if hasattr(model, "model"):
      model = model.model

    # Disable cache globally to avoid graph breaks and recompiles in per-layer compile.
    # This test does not want to enable this.
    model.config.use_cache = False
    example_inputs = dict(model_and_input.example_inputs)
    example_inputs.pop("past_key_values", None)

    return model_utils.ModelAndInput(model=model, example_inputs=example_inputs)

  def _run_benchmark(self, benchmark_name, benchmark_fn, warmup_fn=None):
    warmup_func = warmup_fn if warmup_fn is not None else benchmark_fn

    # Measure warmup
    warmup_start_time = time.time()
    first_step_start = 0.0
    first_step_time_seconds = 0.0

    for step in range(benchmark_utils.MIN_WARMUP_STEPS.value):
      if step == 0:
        first_step_start = time.time()
      with traceme.TraceMe("Warmup", step_num=step):
        warmup_func(step)
      if step == 0:
        first_step_time_seconds = time.time() - first_step_start
    torch.tpu.synchronize()
    warmup_time = time.time() - warmup_start_time

    with benchmark_utils.XprofContext("Eval", True) as xprof_context:
      loop_start_time = time.time()
      for step in range(benchmark_utils.POST_WARMUP_STEPS.value):
        with traceme.TraceMe("Eval", step_num=step):
          benchmark_fn(step)
      torch.tpu.synchronize()
      loop_end_time = time.time()

    post_warmup_step_time_seconds = (
        (loop_end_time - loop_start_time)
    ) / benchmark_utils.POST_WARMUP_STEPS.value

    # Calculate warmup overhead
    warmup_overhead_seconds = (first_step_time_seconds + warmup_time) - (
        post_warmup_step_time_seconds * benchmark_utils.MIN_WARMUP_STEPS.value
    )

    session_id = xprof_context.session_id
    xprof_client = performance_utils.get_xprof_client()
    peak_device_memory_mb = device_utils.get_peak_memory_hbm(
        "tpu", session_id, xprof_client
    )

    self._write_benchmark_result(
        benchmark_name,
        session_id,
        post_warmup_step_time_seconds=post_warmup_step_time_seconds,
        first_step_time_seconds=first_step_time_seconds,
        warmup_overhead_seconds=warmup_overhead_seconds,
        peak_device_memory_mb=peak_device_memory_mb,
    )
    return session_id

  def _write_benchmark_result(
      self,
      benchmark_name,
      session_id,
      post_warmup_step_time_seconds,
      first_step_time_seconds,
      warmup_overhead_seconds,
      peak_device_memory_mb,
  ):
    try:
      duty_cycle = _get_tpu_duty_cycle(session_id)
      duty_str = f"{duty_cycle:.2f}%"
    except Exception:  # pylint: disable=broad-except
      duty_str = "N/A"

    xprof_url = f"http://xprof/?session_id={session_id}"
    log_msg = (
        f"{benchmark_name}: took {post_warmup_step_time_seconds * 1000:.2f} ms"
        f" per step, TPU Duty Cycle: {duty_str}, XProf URL: {xprof_url}"
    )
    logging.info(log_msg)
    self.logs.append(log_msg)

    rank = int(os.environ.get("RANK", "0"))
    if benchmark_utils.MLCOMPASS_TRACKING_ID.value and rank == 0:
      metrics = benchmark_utils.PerformanceBenchmarkResult(
          num_warmup_steps=benchmark_utils.MIN_WARMUP_STEPS.value,
          first_step_time_seconds=first_step_time_seconds,
          warmup_overhead_seconds=warmup_overhead_seconds,
          post_warmup_step_time_seconds=post_warmup_step_time_seconds,
          peak_device_memory_mb=peak_device_memory_mb,
      )
      mlcompass_utils.export_to_mlcompass(
          platform=benchmark_utils.PLATFORM.value,
          metrics=metrics,
          base_cl=benchmark_utils.BASE_CL.value,
          mlcompass_tracking_id=benchmark_utils.MLCOMPASS_TRACKING_ID.value,
          mlcompass_execution_mode=benchmark_utils.MLCOMPASS_EXECUTION_MODE.value,
          test_method_name=self._testMethodName,
          benchmark_name=benchmark_name,
      )

  def test_eager_forward(self):
    device = torch.device("tpu")
    model_and_input = self._get_qwen3_1_7b(device)
    model = model_and_input.model
    example_inputs = model_and_input.example_inputs

    def benchmark_fn(_):
      out = model(**example_inputs)
      x = out.last_hidden_state.cpu()
      del x

    self._run_benchmark("hf_qwen3_1_7b_eager", benchmark_fn)

  def test_compile_forward(self):
    device = torch.device("tpu")
    model_and_input = self._get_qwen3_1_7b(device)
    model = model_and_input.model
    example_inputs = model_and_input.example_inputs

    compiled_model = torch.compile(
        model,
        backend="tpu",
        dynamic=False,
        fullgraph=True,
    )

    def benchmark_fn(_):
      out = compiled_model(**example_inputs)
      x = out.last_hidden_state.cpu()
      del x

    self._run_benchmark("hf_qwen3_1_7b_compile", benchmark_fn)

  def test_per_layer_compile_forward(self):
    device = torch.device("tpu")
    model_and_input = self._get_qwen3_1_7b(device)
    model = model_and_input.model
    example_inputs = model_and_input.example_inputs

    for i in range(len(model.layers)):
      model.layers[i] = torch.compile(
          model.layers[i],
          backend="tpu",
          dynamic=False,
          fullgraph=True,
      )

    def benchmark_fn(_):
      out = model(**example_inputs)
      x = out.last_hidden_state.cpu()
      del x

    with torch.inference_mode():
      self._run_benchmark("hf_qwen3_1_7b_per_layer_compile", benchmark_fn)

  def test_compile_with_async_copy(self):
    device = torch.device("tpu")
    model_and_input = self._get_qwen3_1_7b(device)
    model = model_and_input.model
    example_inputs = model_and_input.example_inputs

    compiled_model = torch.compile(
        model,
        backend="tpu",
        dynamic=False,
        fullgraph=True,
    )

    # We use fixed BS=1 and Seq=256 as per helper
    batch_size = 1
    seq_len = 256

    pinned_out = torch.empty(
        batch_size,
        seq_len,
        model.config.hidden_size,
        dtype=torch.bfloat16,
        device="cpu",
        pin_memory=True,
    )

    # Async Copy Strategy:
    # 1. The main thread exclusively handles dispatching the TPU operations
    #     to the device.
    # 2. The background thread exclusively handles the Device-to-Host (D2H)
    #     transfer via `.copy_()`.
    # 3. Using pinned memory avoids an extra host-side buffer copy, enabling
    #    direct DMA offload.
    def copy_fn(out):
      pinned_out.copy_(out.last_hidden_state, non_blocking=False)

    with torch.inference_mode():
      with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:

        def warmup_fn(_):
          out = compiled_model(**example_inputs)
          executor.submit(copy_fn, out)

        def benchmark_fn(_):
          out = compiled_model(**example_inputs)
          executor.submit(copy_fn, out)

        self._run_benchmark(
            "hf_qwen3_1_7b_compile_async_copy", benchmark_fn, warmup_fn
        )

  def test_compile_with_synced_copy(self):
    device = torch.device("tpu")
    model_and_input = self._get_qwen3_1_7b(device)
    model = model_and_input.model
    example_inputs = model_and_input.example_inputs

    compiled_model = torch.compile(
        model,
        backend="tpu",
        dynamic=False,
        fullgraph=True,
    )

    # We use fixed BS=1 and Seq=256 as per helper
    batch_size = 1
    seq_len = 256

    pinned_out = torch.empty(
        batch_size,
        seq_len,
        model.config.hidden_size,
        dtype=torch.bfloat16,
        device="cpu",
        pin_memory=True,
    )

    def warmup_fn(_):
      out = compiled_model(**example_inputs)
      pinned_out.copy_(out.last_hidden_state)
      del out

    def benchmark_fn(_):
      out = compiled_model(**example_inputs)
      pinned_out.copy_(out.last_hidden_state)
      del out

    with torch.inference_mode():
      self._run_benchmark(
          "hf_qwen3_1_7b_compile_synced_copy", benchmark_fn, warmup_fn
      )


if __name__ == "__main__":
  absltest.main()
