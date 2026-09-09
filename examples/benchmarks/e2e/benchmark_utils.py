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

"""Utilities for running end-to-end benchmarks."""

import contextlib
import dataclasses
import enum
import time
from typing import Any, Callable, Iterable, Sequence

from absl import flags
from absl import logging
import numpy as np
import torch
from torch.utils import _pytree as pytree
from torch_tpu._internal.utils import log_utils
from examples.benchmarks.e2e import device_utils
from examples.benchmarks.e2e.harness import metrics as metrics_lib
from examples.benchmarks.quality_utils import quality_benchmark_model
from torch_tpu._internal.benchmarks import xprof_adapter

log_utils.log_to_stderr()


MAX_WARMUP_STEPS = flags.DEFINE_integer(
    "max_warmup_steps", 20, "Maximum number of warmup steps."
)
MIN_WARMUP_STEPS = flags.DEFINE_integer(
    "min_warmup_steps", 10, "Minimum number of warmup steps."
)
POST_WARMUP_STEPS = flags.DEFINE_integer(
    "post_warmup_steps", 10, "Number of post-warmup steps."
)
XPROF_STEPS = flags.DEFINE_integer(
    "xprof_steps", 2, "Number of steps to collect xprof for."
)
SMOKE_TEST = flags.DEFINE_bool(
    "smoke_test", False, "Whether to run in smoke test mode."
)


def _do_post_warmup() -> bool:
  return POST_WARMUP_STEPS.value > 0


# Used to disable some of the checks if the user
# just wants a single warmup run.
def _is_warmup_only() -> bool:
  return (
      MIN_WARMUP_STEPS.value == 1
      and POST_WARMUP_STEPS.value == 0
      and MAX_WARMUP_STEPS.value == 1
  )


BASE_CL = flags.DEFINE_string(
    "base_cl",
    None,
    "Base CL used for the benchmark run. This needs to be attached to the data"
    " exported to MLCompass.",
)

PENDING_CL = flags.DEFINE_string(
    "pending_cl",
    None,
    "Pending CL used for the benchmark run.",
)

BENCHMARK_GROUP = flags.DEFINE_enum(
    "benchmark_group",
    "experiment",
    ["control", "experiment"],
    "Benchmark group (control or experiment).",
)

TENSORBOARD_OUTPUT_ENV_VAR = flags.DEFINE_string(
    "tensorboard_output_env_var",
    "TENSORBOARD_OUTPUT_DIR",
    "Environment variable to use to retrieve TensorBoard output directory.",
)

ENABLE_TENSORBOARD_LOGGING = flags.DEFINE_bool(
    "enable_tensorboard_logging",
    False,
    "Whether to enable TensorBoard logging for the benchmark results.",
)


MLCOMPASS_TRACKING_ID = flags.DEFINE_string(
    "mlcompass_tracking_id",
    None,
    "The UUID used to track all the generated metrics by a single invocation."
    " If provided, the metrics will be exported to MLCompass DB. The flag is"
    " optional and the user should only set it if they want to manually trigger"
    " an export to MLCompass. Refer to"
    " go/exporting-to-mlcompass#mlcompass-tracking-id for more details.",
)

MLCOMPASS_EXECUTION_MODE = flags.DEFINE_enum(
    "mlcompass_execution_mode",
    "oneshot",
    ["cbuild", "oneshot", "cbuild-autoperfcop", "autoperfcop"],
    "The execution mode of the mlcompass run. This is only used to filter out"
    " data for dashboarding post run and doesn't have any impact on the run."
    " The value is set to cbuild for continuous guitar runs and oneshot"
    " otherwise. Only cbuild runs' results are recorded in the dashboards.",
)

# This limit is enough to measure two steps for fsdp benchmark which is
# currently the longest running benchmark. Without the limit, the
# process uploading profile can fail health checks and die.
_PROFILE_MAX_BYTES = 500 * 1024 * 1024  # 500 MB


class BenchmarkCategory(enum.Enum):
  """The category of the benchmark."""

  HUGGINGFACE_LLM = "huggingface_llm"
  HUGGINGFACE_DIFFUSER = "huggingface_diffuser"
  HUGGINGFACE_VISION = "huggingface_vision"
  META_LLAMA = "meta_llama"
  ML_LAYER = "ml_layer"
  TIMM = "timm"
  QWEN_RAGGED_MOE = "qwen_ragged_moe"
  QWEN3_5_MOE = "qwen3_5_moe"
  GEMMA_RAGGED_MOE = "gemma_ragged_moe"
  TORCHAUDIO = "torchaudio"
  INTERNAL_MODEL = "internal_model"


class XprofContext:
  """A context manager for Xprof sessions that logs the session URL.

  Attributes:
    name: A name for the xprof session, used in logging.
    enable_xprof: Whether to enable xprof profiling. If False, the context
      manager does nothing.
    trace_only_host: Whether to trace only the host.
    session: The xprof_adapter.XprofSession object if xprof is enabled,
      otherwise None.
    session_id: The ID of the xprof session after it has ended, if xprof was
      enabled. Otherwise None.
  """

  def __init__(
      self, name: str, enable_xprof: bool, trace_only_host: bool = False
  ):
    self.name = name
    self.enable_xprof = enable_xprof
    self.trace_only_host = trace_only_host
    self.session = None
    self.session_id = None

  def __enter__(self):
    if self.enable_xprof:
      self.session = xprof_adapter.XprofSession()
      kwargs = {}
      host_trace_level = 3
      enable_python_tracer = True
      if self.trace_only_host:
        kwargs["trace_mode"] = "TRACE_ONLY_HOST"
        host_trace_level = 1
        enable_python_tracer = False
      self.session.start_session(
          host_trace_level=host_trace_level,
          enable_python_tracer=enable_python_tracer,
          host_cpu_profile=True,
          response_max_bytes=_PROFILE_MAX_BYTES,
          **kwargs,
      )
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    # End the session and log the URL if it was started. If there is an
    # exception, we let the exception propagate.
    if self.session:
      self.session_id = self.session.end_session_and_get_session_id()
      logging.info(
          "%s xprof URL: http://xprof/?session_id=%s",
          self.name,
          self.session_id,
      )


def _get_device_name(device: torch.device) -> str:
  """Returns the device name from a torch.device object.

  cuda:0 and tpu:0 will return cuda and tpu respectively.
  """

  device_str = str(device)
  device_name, *_ = device_str.split(":", maxsplit=1)
  return device_name


def _get_warmup_overhead(timings: np.ndarray, num_warmup_steps: int) -> float:
  """Calculates the warmup overhead in seconds from the timings of the warmup runs."""

  if _is_warmup_only():
    return timings[0]

  if not num_warmup_steps:
    raise RuntimeError(
        "Benchmark function compilations have not stabilized after"
        f" {MAX_WARMUP_STEPS.value} warmup runs. num_warmup_steps was"
        f" {num_warmup_steps}. Consider increasing the number of warmup steps."
    )

  return (
      np.sum(timings[:num_warmup_steps])
      - timings[num_warmup_steps] * num_warmup_steps
  )


def _run_step(
    benchmark_function: Callable[
        [torch.nn.Module, Any, torch.optim.Optimizer | None],
        Any,
    ],
    model: torch.nn.Module,
    step_input: Any,
    optimizer: torch.optim.Optimizer | None,
    device_name: str,
    sync_params: bool,
) -> Any:
  """Runs a single benchmark step and synchronizes the device.

  Args:
    benchmark_function: The benchmark function to run.
    model: The model to run the step on.
    step_input: The inputs to pass to the benchmark function.
    optimizer: The optimizer to use for the step, or None.
    device_name: The name of the device (e.g. 'tpu', 'cuda').
    sync_params: Whether to eagerly synchronize parameter gradients.

  Returns:
    The output of the benchmark function.
  """
  out = benchmark_function(model, step_input, optimizer)
  device_utils.synchronize(device_name, out)
  if sync_params:
    for p in model.parameters():
      if p.grad is not None:
        device_utils.synchronize(device_name, p.grad)
  return out


def _warmup_run(
    benchmark_function: Callable[
        [torch.nn.Module, Any, torch.optim.Optimizer | None],
        Any,
    ],
    model: torch.nn.Module,
    example_inputs: Any,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    enable_xprof: bool = False,
    sync_params: bool = False,
) -> metrics_lib.WarmupRunResult:
  """Runs the model for MAX_WARMUP_STEPS times to warmup the caches.

  If the benchmark function compilations have not stabilized after
  MAX_WARMUP_STEPS runs, then the warmup result is not calculated and
  RuntimeError is raised.

  Args:
    benchmark_function: The benchmark function to run.
    model: The model to run.
    example_inputs: The example inputs to run the model with.
    device: The device to run the benchmark on.
    optimizer: The optimizer to use for the model.
    enable_xprof: Whether to enable xprof profiling.
    sync_params: Whether to eagerly synchronize parameter gradients inside
      timing loops.

  Returns:
    A WarmupRunResult instance containing the number of warmup steps, the time
    taken for the first step, and the warmup overhead in seconds.
  """
  # TODO(bbahl): Decide the number of warmup steps dynamically, possibly based
  # on cache miss count.
  timings = np.zeros(MAX_WARMUP_STEPS.value, dtype=np.float64)
  # cache misses is always 0 for CUDA. In this case,
  # we just use the first run as the preheat overhead.
  cache_misses = np.zeros(MAX_WARMUP_STEPS.value, dtype=np.int64)
  num_warmup_steps = None
  device_name = _get_device_name(device)

  is_sequence = isinstance(example_inputs, list) and len(example_inputs) > 0

  with XprofContext(
      "warmup_run", enable_xprof, trace_only_host=True
  ) as warmup_run_context:
    for step in range(MAX_WARMUP_STEPS.value):
      with xprof_adapter.TraceMe("Warmup", step_num=step):
        step_input = example_inputs[step] if is_sequence else example_inputs
        start_time = time.perf_counter()
        _run_step(
            benchmark_function,
            model,
            step_input,
            optimizer,
            device_name,
            sync_params,
        )
        timings[step] = time.perf_counter() - start_time
      cache_misses[step] = device_utils.cache_miss_count(device_name)

      if (
          step >= MIN_WARMUP_STEPS.value
          and cache_misses[step] == cache_misses[step - 1]
      ):
        num_warmup_steps = step
        break

  logging.info("Warmup Timings: %s", timings)
  logging.info("Warmup cache misses: %s", cache_misses)
  warmup_session_xprof_url = None
  if enable_xprof:
    warmup_session_xprof_url = (
        f"http://xprof/?session_id={warmup_run_context.session_id}"
    )

  return metrics_lib.WarmupRunResult(
      num_warmup_steps=num_warmup_steps,  # pyrefly: ignore[bad-argument-type]
      first_step_time_seconds=timings[0],
      warmup_overhead_seconds=_get_warmup_overhead(timings, num_warmup_steps),  # pyrefly: ignore[bad-argument-type]
      warmup_session_xprof_url=warmup_session_xprof_url,
  )


def _post_warmup_run(
    benchmark_function: Callable[
        [torch.nn.Module, Any, torch.optim.Optimizer | None],
        Any,
    ],
    model: torch.nn.Module,
    example_inputs: Any,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    enable_xprof: bool = False,
    xprof_client: xprof_adapter.XprofAnalysisClient | None = None,
    sync_params: bool = False,
) -> metrics_lib.PostWarmupRunResult:
  """Runs the model once after the warmup is complete.

  Args:
    benchmark_function: The benchmark function to run.
    model: The model to run.
    example_inputs: The example inputs to run the model with.
    device: The device the model and input data is on.
    optimizer: The optimizer to use for the model.
    enable_xprof: Whether to enable xprof profiling.
    xprof_client: The xprof client to use for profiling.
    sync_params: Whether to eagerly synchronize parameter gradients inside
      timing loops.

  Returns:
    A PostWarmupRunResult instance containing the average step time and peak
    device memory usage.
  """

  non_xprof_timings = []
  xprof_timings = []
  device_utils.reset_peak_memory_stats(_get_device_name(device))
  num_cache_misses = None
  device_name = _get_device_name(device)

  is_sequence = isinstance(example_inputs, list) and len(example_inputs) > 0

  # TODO(bbahl): Calculate the number of post warmup steps based on timing
  # information.
  xprof_steps = (
      min(POST_WARMUP_STEPS.value, XPROF_STEPS.value) if enable_xprof else 0
  )
  xprof_context = None

  def run_single_step(step, is_profiling_run: bool):
    nonlocal num_cache_misses, xprof_timings, non_xprof_timings
    ctx = (
        xprof_adapter.TraceMe("Eval", step_num=step)
        if is_profiling_run
        else contextlib.nullcontext()
    )
    with ctx:
      step_input = example_inputs[step] if is_sequence else example_inputs
      start_time = time.perf_counter()
      _run_step(
          benchmark_function,
          model,
          step_input,
          optimizer,
          device_name,
          sync_params,
      )
      elapsed = time.perf_counter() - start_time
      if is_profiling_run:
        xprof_timings.append(elapsed)
      else:
        non_xprof_timings.append(elapsed)

    # Assert that the cache misses are consistent across steps.
    if not is_sequence:
      # TODO(unda): Re-introduce a version of this check for bounded dynamism.
      step_cache_misses = device_utils.cache_miss_count(device_name)
      if num_cache_misses is None:
        num_cache_misses = step_cache_misses
      if step_cache_misses != num_cache_misses:
        raise RuntimeError(
            "Cache misses are not consistent across steps; expected"
            f" {num_cache_misses}, got {step_cache_misses}. This means that"
            " the model is not fully warmed up after"
            f" {MAX_WARMUP_STEPS.value} warmup steps. Consider increasing"
            " the number of warmup steps."
        )

  if xprof_steps > 0:
    with XprofContext("post_warmup_run", True) as xprof_context:
      for step in range(xprof_steps):
        run_single_step(step, is_profiling_run=True)

  for step in range(xprof_steps, POST_WARMUP_STEPS.value):
    run_single_step(step, is_profiling_run=False)

  post_warmup_run_session_xprof_url = None
  if xprof_context and xprof_context.session_id:
    post_warmup_run_session_xprof_url = (
        f"http://xprof/?session_id={xprof_context.session_id}"
    )

  # Calculate the memory usage after the post warmup run is complete. This
  # requires the xprof response for TPU and XLA_CUDA devices, which is only
  # available after the xprof session ends. Memory usage is calculated for TPU
  # and XLA_CUDA devices only when Xprof is enabled.
  session_id = xprof_context.session_id if xprof_context else None
  memory_usage = device_utils.get_peak_memory_hbm(
      device_name, session_id, xprof_client
  )

  total_device_time = device_utils.get_max_total_device_time(
      session_id, xprof_client
  )

  avg_device_time = -1.0
  if total_device_time != -1.0 and xprof_steps > 0:
    avg_device_time = total_device_time / xprof_steps

  logging.info("Post Warmup Timings (non-xprof): %s", non_xprof_timings)
  logging.info("Post Warmup Timings (xprof): %s", xprof_timings)

  if not non_xprof_timings:
    raise ValueError(
        "non_xprof_timings is empty. There must be at least one non-profiling"
        " step to measure post warmup performance."
    )

  return metrics_lib.PostWarmupRunResult(
      post_warmup_step_time_seconds=np.mean(non_xprof_timings),
      peak_device_memory_mb=memory_usage,
      post_warmup_run_session_xprof_url=post_warmup_run_session_xprof_url,
      average_post_warmup_device_time_seconds=avg_device_time,
  )


def _smoke_test_run(
    benchmark_function: Callable[
        [torch.nn.Module, Any, torch.optim.Optimizer | None],
        Any,
    ],
    model: torch.nn.Module,
    example_inputs: Any,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    sync_params: bool = False,
) -> None:
  """Runs the model for a single step (or multiple steps for dynamism tests).

  This function only checks if the model runs successfully and doesn't measure
  any metrics for model runs.

  Args:
    benchmark_function: The benchmark function to run.
    model: The model to run.
    example_inputs: The example inputs to run the model with.
    device: The device the model and input data is on.
    optimizer: The optimizer to use for the model. Needed for training.
    sync_params: Whether to eagerly synchronize parameter gradients inside
      timing loops.
  """
  device_name = _get_device_name(device)
  is_sequence = isinstance(example_inputs, list) and len(example_inputs) > 0
  num_steps = len(example_inputs) if is_sequence else 1

  for step in range(num_steps):
    step_input = example_inputs[step] if is_sequence else example_inputs
    _run_step(
        benchmark_function,
        model,
        step_input,
        optimizer,
        device_name,
        sync_params,
    )


def _synchronize_all_tensors(tensor_pytree: Any, device: torch.device):
  """Synchronizes example inputs to device."""

  def _sync_element(elem):
    if isinstance(elem, torch.Tensor):
      device_utils.synchronize(device.type, elem)
    return elem

  pytree.tree_map(_sync_element, tensor_pytree)


def run_performance_benchmark(
    benchmark_function: Callable[
        [torch.nn.Module, Any, torch.optim.Optimizer | None],
        Any,
    ],
    model: torch.nn.Module,
    example_inputs: Any,
    device: torch.device,
    *,
    enable_xprof: bool = False,
    optimizer: torch.optim.Optimizer | None = None,
    xprof_client: xprof_adapter.XprofAnalysisClient | None = None,
    sync_params: bool = False,
    is_bounded_dynamic: bool = False,
) -> metrics_lib.PerformanceMetrics:
  """Runs a performance benchmark for a given model.

  Args:
    benchmark_function: The benchmark function to run. This function is expected
      to take the model, example inputs and optimizer as arguments. If the
      return value is a torch.Tensor, we synchronize on the tensor to calculate
      the performance metrics.
    model: The model to run.
    example_inputs: The example inputs to run the model with.
    device: The device the model and input data is on.
    enable_xprof: Whether to enable xprof profiling.
    optimizer: The optimizer to use for the model. Needed for training
      benchmarks.
    xprof_client: The xprof client to use for profiling.
    is_bounded_dynamic: Whether the example inputs are bounded dynamic.

  Returns:
    A PerformanceMetrics instance containing the results of the
    benchmark.
  """
  if _get_device_name(device) == "cuda" and not torch.cuda.is_available():
    logging.warning("CUDA is not available. Skipping CUDA benchmark.")
    # Return a default result if CUDA is not available to avoid failing
    # presubmits on non-CUDA environments.
    return metrics_lib.PerformanceMetrics()

  _synchronize_all_tensors(example_inputs, device)
  _synchronize_all_tensors(list(model.state_dict().values()), device)

  if SMOKE_TEST.value:
    _smoke_test_run(
        benchmark_function,
        model,
        example_inputs,
        device,
        optimizer=optimizer,
        sync_params=sync_params,
    )
    return metrics_lib.PerformanceMetrics()

  warmup_steps = MAX_WARMUP_STEPS.value
  if is_bounded_dynamic and (
      not isinstance(example_inputs, list)
      or len(example_inputs) != (warmup_steps + POST_WARMUP_STEPS.value)
  ):
    logging.warning(
        "Example inputs are not a list of length warmup_steps +"
        " POST_WARMUP_STEPS. This is unexpected for bounded dynamic"
        " benchmarks. Skipping benchmark."
    )
    return metrics_lib.PerformanceMetrics()

  warmup_inputs = (
      example_inputs[:warmup_steps] if is_bounded_dynamic else example_inputs
  )
  post_warmup_inputs = (
      example_inputs[warmup_steps:] if is_bounded_dynamic else example_inputs
  )

  result_kwargs = {}
  start_time = time.perf_counter()
  result_kwargs |= dataclasses.asdict(
      _warmup_run(
          benchmark_function,
          model,
          warmup_inputs,
          device,
          optimizer=optimizer,
          enable_xprof=enable_xprof,
          sync_params=sync_params,
      )
  )

  if _do_post_warmup():
    result_kwargs |= dataclasses.asdict(
        _post_warmup_run(
            benchmark_function,
            model,
            post_warmup_inputs,
            device,
            optimizer=optimizer,
            enable_xprof=enable_xprof,
            xprof_client=xprof_client,
            sync_params=sync_params,
        )
    )

  result_kwargs["peak_host_compilation_memory_mb"] = (
      device_utils.get_peak_host_compilation_memory_mb(_get_device_name(device))
  )

  return metrics_lib.PerformanceMetrics(
      e2e_wall_time_seconds=time.perf_counter() - start_time,
      **result_kwargs,
  )


def run_quality_benchmark(
    benchmark_model: quality_benchmark_model.QualityBenchmarkModel,
    model_compile: bool,
    dataset_loader: Iterable[Any],
    benchmark_metrics: Sequence[quality_benchmark_model.MetricProducer],
    device: torch.device,
) -> metrics_lib.QualityMetrics:
  """Runs a quality benchmark for a given model.

  Args:
    benchmark_model: The benchmark model object.
    model_compile: Whether to compile the model.
    dataset_loader: A function that loads the dataset.
    benchmark_metrics: A list of metric producers to run.
    device: The device the model and input data is on.

  Returns:
    A QualityMetrics instance containing the results of the benchmark.
  """
  benchmark_model.initialize()
  start_time = time.perf_counter()
  metric_scores = {}
  for benchmark_metric in benchmark_metrics:
    score = quality_benchmark_model.run_quality_benchmark(
        benchmark_model, model_compile, dataset_loader, benchmark_metric
    )
    # Add metric to score map. Furthermore, it materializes the score.
    metric_scores[benchmark_metric.get_name()] = score.item()  # pyrefly: ignore[missing-attribute]
  logging.info("Metric scores (device: %s): %s", device, metric_scores)

  return metrics_lib.QualityMetrics(
      e2e_wall_time_seconds=time.perf_counter() - start_time,
      metrics=metric_scores,
  )
