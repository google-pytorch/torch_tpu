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
import dataclasses
import json
import logging
import os
import sys

# Ensure the repo root is in sys.path first to avoid import shadowing of 'examples' package in CI.
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_OSS_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", "..", ".."))
_G3_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", "..", "..", ".."))
if _OSS_ROOT not in sys.path:
  sys.path.insert(0, _OSS_ROOT)
if _G3_ROOT not in sys.path:
  sys.path.insert(0, _G3_ROOT)

import time
from typing import Any, Callable, Dict, List, Mapping, Tuple
import unittest
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
import torch.utils.benchmark as benchmark
from examples.benchmarks.e2e import device_utils
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import common
from examples.benchmarks.e2e import mlcompass_utils
from examples.benchmarks.e2e.harness import metrics as metrics_lib
from examples.benchmarks.ops.op_input_loader import deserialize_args

DEVICE = flags.DEFINE_string("device", "tpu", "Device to run benchmarks on.")
MIN_RUN_TIME = flags.DEFINE_float(
    "min_run_time", 2.0, "Minimum run time for blocked_autorange."
)


@dataclasses.dataclass
class OpBenchmarkResult(metrics_lib.MetricsInterface):
  """Result of a single operator benchmark run."""

  median_time_us: float = 0.0
  compile_time_s: float = 0.0
  cv: float = 0.0
  source_metrics: Mapping[str, Mapping[str, float]] = dataclasses.field(
      default_factory=dict
  )

  def metric_map(self) -> Mapping[str, float]:
    """Returns a map of metrics to be exported to MLCompass."""
    metrics = {
        "median_time_us": self.median_time_us,
        "compile_time_s": self.compile_time_s,
        "cv": self.cv,
    }
    for source, data in self.source_metrics.items():
      # Clean up source name to be a valid metric name
      clean_source = source.replace(".", "_").replace("-", "_")
      metrics[f"count_{clean_source}"] = float(data.get("count", 0.0))
      metrics[f"weight_{clean_source}"] = float(data.get("weight", 0.0))
    return metrics


def get_test_cases() -> (
    List[Tuple[str, str, int, str, str, Mapping[str, Mapping[str, float]]]]
):
  """Generates test cases by reading the captured operator JSON file(s).

  Supports multiple comma-separated files in CAPTURE_FILE. Supports TOP_N
  filtering via environment variable.
  """
  capture_file_str = os.environ.get("CAPTURE_FILE")
  script_dir = os.path.dirname(os.path.abspath(__file__))
  captured_ops_dir = os.path.join(script_dir, "captured_ops")
  import glob

  if not capture_file_str:
    if os.path.exists(captured_ops_dir):
      capture_files = sorted(
          glob.glob(os.path.join(captured_ops_dir, "*.json"))
      )
      logging.info(
          f"Defaulting to all JSON files in {captured_ops_dir}: {capture_files}"
      )
    else:
      capture_files = [os.path.join(script_dir, "minimal_ops.json")]
  else:
    capture_files = sorted([f.strip() for f in capture_file_str.split(",")])

  from collections import defaultdict

  aggregated_data = defaultdict(lambda: defaultdict(dict))
  file_total_counts = defaultdict(int)

  for cf in capture_files:
    if not os.path.exists(cf):
      logging.warning(f"Capture file {cf} not found. Skipping.")
      continue
    try:
      with open(cf, "r") as f:
        data = json.load(f)
        # Sort keys of data to ensure deterministic op order processing
        for op_name in sorted(data.keys()):
          if op_name == "CLUSTER":
            continue
          items = data[op_name]
          for item in items:
            inputs_str = item["inputs"]
            count = item.get("count", 1)
            # Use basename of file as source key, stripped of common suffix
            source_key = os.path.basename(cf).replace(
                "_eager_default_ops.json", ""
            )
            aggregated_data[op_name][inputs_str][source_key] = count
            file_total_counts[source_key] += count
    except Exception as e:
      logging.error(f"Failed to load capture file {cf}: {e}")

  # Calculate total frequency for each operator across all files stable-sorted
  op_frequencies = {}
  for op_name in sorted(aggregated_data.keys()):
    inputs_map = aggregated_data[op_name]
    total_count = 0
    for inputs_str in sorted(inputs_map.keys()):
      sources = inputs_map[inputs_str]
      total_count += sum(sources[src] for src in sorted(sources.keys()))
    op_frequencies[op_name] = total_count

  # Filter by TOP_N and sort deterministically
  top_n = int(os.environ.get("TOP_N", "0"))  # 0 means all
  if top_n > 0:
    # Sort by frequency (descending), and primary by op_name (ascending) for stability
    sorted_ops = sorted(op_frequencies.items(), key=lambda x: (-x[1], x[0]))[
        :top_n
    ]
    target_op_names = [op[0] for op in sorted_ops]
    logging.info(f"Filtering to Top {top_n} operators: {target_op_names}")
  else:
    target_op_names = sorted(aggregated_data.keys())

  cases = []
  for op_name in target_op_names:
    inputs_map = aggregated_data[op_name]
    sorted_inputs = sorted(inputs_map.keys())
    for i, inputs_str in enumerate(sorted_inputs):
      sources = inputs_map[inputs_str]
      for run_mode_str in sorted(["eager", "compiled"]):
        # Create a safe test name
        safe_op_name = op_name.replace(".", "_").replace(":", "_")
        test_name = f"{safe_op_name}_case{i}_{run_mode_str}"

        source_metrics = {}
        for source in sorted(sources.keys()):
          count = sources[source]
          total = file_total_counts[source]
          weight = count / total if total > 0 else 0.0
          source_metrics[source] = {"count": float(count), "weight": weight}

        cases.append(
            (test_name, op_name, i, inputs_str, run_mode_str, source_metrics)
        )
  # Sort cases by test name to guarantee a perfectly stable list across shards
  cases.sort(key=lambda x: x[0])
  return cases


@contextlib.contextmanager
def _aten_run_mode_context(run_mode_str: str):
  """Self-contained context manager to configure execution mode for ATen op benchmarks."""
  if run_mode_str == "compiled":
    try:
      yield
    finally:
      if hasattr(torch, "_dynamo"):
        # pylint: disable=protected-access
        torch._dynamo.reset()
    return

  original_eager_mode = None
  execution_mode = None
  # torch_tpu._internal.execution_mode is only available when torch_tpu is
  # installed. When benchmarking on non-TPU backends (e.g., CPU or CUDA with
  # vanilla PyTorch), execution_mode is not importable; ignoring ImportError
  # is acceptable because there is no TPU eager mode to configure.
  try:
    from torch_tpu._internal import execution_mode  # pylint: disable=g-import-not-at-top
  except ImportError:
    pass
  else:
    original_eager_mode = execution_mode.eager_mode
    execution_mode.eager_mode = execution_mode.EagerMode.DEFER_NEVER

  try:
    yield
  finally:
    if original_eager_mode is not None and execution_mode is not None:
      execution_mode.eager_mode = original_eager_mode


class AtenOpBenchmarkBase(parameterized.TestCase):

  def _resolve_op(self, op_name: str) -> Callable | None:
    """Resolves an operator name string to a callable PyTorch operation."""
    parts = op_name.split(".")
    if len(parts) >= 2 and parts[0] == "aten":
      try:
        op_ns = getattr(torch.ops, parts[0])
        op_overload = getattr(op_ns, parts[1])
        if len(parts) > 2:
          op_overload = getattr(op_overload, parts[2])
        return op_overload
      except AttributeError:
        logging.warning(f"Could not resolve op: {op_name}")
        return None
    return None

  def _run_benchmark(
      self,
      op_name: str,
      case_index: int,
      inputs_str: str,
      run_mode_str: str,
      device_type: str,
      backend_name: str,
      source_metrics: Mapping[str, Mapping[str, float]],
  ):
    """Internal method to run the benchmark."""
    device = torch.device(device_type)

    try:
      args, kwargs = deserialize_args(inputs_str, device=device_type)
    except Exception as e:
      self.skipTest(f"Failed to deserialize inputs: {e}")

    op_callable = self._resolve_op(op_name)
    if op_callable is None:
      self.skipTest(f"Could not resolve operator: {op_name}")

    if device_type == "cpu" and op_name == "aten.mm.default":
      try:
        args, kwargs = deserialize_args(inputs_str, device="cpu")
        if (
            len(args) >= 2
            and isinstance(args[0], torch.Tensor)
            and isinstance(args[1], torch.Tensor)
        ):
          m, k = args[0].shape
          k2, n = args[1].shape
          flops = m * k * n
          if flops > 1e9:  # 1 Billion flops limit
            self.skipTest(
                f"Skipping large mm on CPU: {m}x{k}x{n} ({flops} flops)"
            )
      except unittest.SkipTest:
        raise
      except Exception as e:
        logging.warning(f"Failed to parse inputs for size check: {e}")

    if (
        device_type == "cpu"
        and op_name
        == "aten._scaled_dot_product_fused_attention_overrideable.default"
    ):
      self.skipTest(f"Skipping {op_name} on CPU (not supported)")

    if (
        backend_name == "torchAX"
        and op_name
        == "aten._scaled_dot_product_fused_attention_overrideable.default"
    ):
      self.skipTest(f"Skipping {op_name} on TorchAX (not supported)")

    run_mode = (
        common.RunMode.COMPILED
        if run_mode_str == "compiled"
        else common.RunMode.EAGER_DEFAULT
    )

    compile_time_s = 0.0
    with _aten_run_mode_context(run_mode_str):
      target_op = op_callable
      if run_mode_str == "compiled":
        start_compile = time.perf_counter()
        if device.type == "cpu":
          # Use aot_eager backend on CPU to avoid missing C++ compiler error
          target_op = torch.compile(op_callable, backend="aot_eager")
        elif device.type == "jax":
          # TorchAX handles compilation differently or not at all via torch.compile
          # For now, we just use the eager callable for TorchAX in compiled mode too
          target_op = op_callable
        else:
          target_op = device_utils.torch_compile(op_callable, device.type)
        compile_time_s = time.perf_counter() - start_compile
        logging.info(f"Compile time for {op_name}: {compile_time_s} s")

      # Wrap target_op to synchronize after each call to prevent TPU queue flooding
      # and ensure the host measures actual device execution times.
      def timed_op(*op_args, **op_kwargs):
        res = target_op(*op_args, **op_kwargs)
        if device.type in ("tpu", "xla_cuda", "cuda"):
          device_utils.synchronize(device.type, res)
        return res

      # Warmup: run only 2 warmup iterations during fast CI/unit-test mode
      # (min_run_time < 0.2s) to conserve runner execution time, while running
      # 20 warmup iterations during precision benchmark mode to ensure PjRt
      # cache and memory allocator stabilization prior to recording measurement
      # times.
      warmup_runs = 2 if MIN_RUN_TIME.value < 0.2 else 20
      for _ in range(warmup_runs):
        timed_op(*args, **kwargs)

      # Timing
      timer = benchmark.Timer(
          stmt="op(*args, **kwargs)",
          globals={"op": timed_op, "args": args, "kwargs": kwargs},
      )

      import gc

      gc.disable()
      try:
        # Use blocked_autorange to enforce minimum run time
        measurement = timer.blocked_autorange(min_run_time=MIN_RUN_TIME.value)
      finally:
        gc.enable()

      # Calculate median and CV using numpy to avoid torch_xla2 quantile error
      times = np.array(measurement.times)
      median_time_us = np.median(times) * 1e6
      mean_time = np.mean(times)
      std_time = np.std(times, ddof=1)
      cv = std_time / mean_time if mean_time > 0 else 0.0

      logging.info(
          f"Result for {op_name} case {case_index} ({run_mode_str}) on"
          f" {device_type} ({backend_name}): {median_time_us} us, CV: {cv}"
      )

      # MLCompass Export
      if benchmark_utils.MLCOMPASS_TRACKING_ID.value:
        result = OpBenchmarkResult(
            median_time_us=median_time_us,
            e2e_wall_time_seconds=sum(measurement.times),
            compile_time_s=compile_time_s,
            cv=cv,
            source_metrics=source_metrics,
        )

        safe_op_name = op_name.replace(".", "_").replace(":", "_")
        mlcompass_utils.export_to_mlcompass(
            platform=common.PLATFORM.value,
            metrics=result,
            base_cl=benchmark_utils.BASE_CL.value,
            mlcompass_tracking_id=benchmark_utils.MLCOMPASS_TRACKING_ID.value,
            mlcompass_execution_mode=benchmark_utils.MLCOMPASS_EXECUTION_MODE.value,
            test_method_name=f"test_{safe_op_name}",
            benchmark_name=f"{backend_name}_{run_mode_str}",
            microbenchmark_name=f"case{case_index}",
            succeeded=True,
            pending_cl=benchmark_utils.PENDING_CL.value,
            benchmark_group=benchmark_utils.BENCHMARK_GROUP.value,
        )


class TorchTpuOpBenchmark(AtenOpBenchmarkBase):

  @parameterized.named_parameters(*get_test_cases())
  def test_op(
      self,
      op_name: str,
      case_index: int,
      inputs_str: str,
      run_mode_str: str,
      source_metrics: Mapping[str, Mapping[str, float]],
  ):
    self._run_benchmark(
        op_name,
        case_index,
        inputs_str,
        run_mode_str,
        DEVICE.value,
        "torchTPU",
        source_metrics,
    )


if __name__ == "__main__":
  absltest.main()
