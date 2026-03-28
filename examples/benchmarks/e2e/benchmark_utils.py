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

import abc
import dataclasses
import enum
import random
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from absl import flags
from absl import logging
import numpy as np
import torch
from torch.utils import _pytree as pytree
from torch_tpu._internal.utils import device_utils
import torch_tpu.api as xla_api
from examples.benchmarks.quality_utils import quality_benchmark_model

from torch_tpu._internal.shims.xprof import xprof_analysis_client
from torch_tpu._internal.shims.xprof import xprof_session


MAX_WARMUP_STEPS = flags.DEFINE_integer(
    "max_warmup_steps", 20, "Maximum number of warmup steps."
)
MIN_WARMUP_STEPS = flags.DEFINE_integer(
    "min_warmup_steps", 10, "Minimum number of warmup steps."
)
POST_WARMUP_STEPS = flags.DEFINE_integer(
    "post_warmup_steps", 10, "Number of post-warmup steps."
)
_RANDOM_SEED = 0


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


class Platform(enum.Enum):
  """The platform to run the benchmark on."""

  # The platform names should match the ones in the MLCompass config file. See
  # go/torchtpu-mlcompass#configuration-structure for more details.
  GFC_1X1X1 = "gfc_1x1x1"
  GFC_2X2X1 = "gfc_2x2x1"
  B200_1 = "b200_1"
  B200_4 = "b200_4"
  B200_8 = "b200_8"


BASE_CL = flags.DEFINE_string(
    "base_cl",
    None,
    "Base CL used for the benchmark run. This needs to be attached to the data"
    " exported to MLCompass.",
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

PLATFORM = flags.DEFINE_enum_class(
    "platform",
    Platform.GFC_1X1X1,
    Platform,
    "The platform to run the tests on.",
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

PLATFORM_DEVICE_MAP = {
    Platform.GFC_1X1X1: "tpu",
    Platform.GFC_2X2X1: "tpu",
    Platform.B200_1: "cuda",
    Platform.B200_4: "cuda",
    Platform.B200_8: "cuda",
}


def seed_rngs() -> None:
  """Seeds the Python and PyTorch RNGs with the given seed."""
  random.seed(_RANDOM_SEED)
  torch.manual_seed(_RANDOM_SEED)


@dataclasses.dataclass
class BenchmarkResultInterface(abc.ABC):
  """Interface for benchmark results.

  Attributes:
    e2e_wall_time_seconds: The total wall time of the benchmark.
  """

  e2e_wall_time_seconds: float = 0.0

  @abc.abstractmethod
  def metric_map(self) -> Mapping[str, float]:
    """Returns a map of metrics to be exported to MLCompass."""
    raise NotImplementedError


class RunMode(enum.Enum):
  """The mode to run the benchmark in.

  Make sure that no entry is a prefix of the other. Run mode name is appended
  to test names, and MLCompass runs tests based on a prefix match, it can lead
  to duplicate entries. For e.g., test_model_eager will match both
  test_model_eager and test_model_eager_optimized. Hence, we use
  eager_default and eager_optimized.
  """

  EAGER_DEFAULT = (  # Run the model in eager mode with O1 XLA optimizations.
      "eager_default"
  )
  EAGER_OPTIMIZED = (  # Run the model in eager mode with O2 XLA optimizations.
      "eager_optimized"
  )
  DEFER_NEVER_ONLY = (  # Run the model in eager mode with DeferNever.
      "defer_never_only"
  )
  DEFER_NEVER_AND_LAUNCH_BLOCKING = (  # Run the model in eager mode with DeferNeverAndLaunchBlocking.
      "defer_never_and_launch_blocking"
  )
  COMPILED = "compiled"  # Run the model with torch.compile.


class BenchmarkCategory(enum.Enum):
  """The category of the benchmark."""

  HUGGINGFACE_LLM = "huggingface_llm"
  META_LLAMA = "meta_llama"
  ML_LAYER = "ml_layer"


@dataclasses.dataclass
class PerformanceBenchmarkResult(BenchmarkResultInterface):
  """Result of a performance benchmark run.

  Attributes:
    num_warmup_steps: The number of warmup steps taken for the cache misses to
      stabilize
    first_step_time_seconds: The time taken for the first step.
    warmup_overhead_seconds: This is the extra time taken to run the benchmark
      to warmup the caches. If it takes n steps for cache misses to stabilize,
      then the warmup overhead is (wall time of n steps) - (wall time of 1 warm
      step) * n. For example, if the cache misses are [100, 120, 130, 130] and
      wall times are [15, 10, 10, 2], then the warmup overhead is (15 + 10 + 10)
      - 2*3 = 29 seconds.
    post_warmup_step_time_seconds: The average run time of a benchmark step
      after the warmup is complete.
    peak_device_memory_mb: The peak device memory usage in MB for a benchmark
      step.
  """

  num_warmup_steps: int = 0
  first_step_time_seconds: float = 0.0
  warmup_overhead_seconds: float = 0.0
  post_warmup_step_time_seconds: float = 0.0
  peak_device_memory_mb: float = 0.0

  def metric_map(self) -> Mapping[str, float]:
    """Returns a map of metrics to be exported to MLCompass."""
    return {
        "num_warmup_steps": self.num_warmup_steps,
        "first_step_time_seconds": self.first_step_time_seconds,
        "warmup_overhead_seconds": self.warmup_overhead_seconds,
        "post_warmup_step_time_seconds": self.post_warmup_step_time_seconds,
        "peak_device_memory_mb": self.peak_device_memory_mb,
    }


@dataclasses.dataclass
class QualityBenchmarkResult(BenchmarkResultInterface):
  """Result of a quality benchmark run.

  Attributes:
    metrics: A map of metrics to be exported to MLCompass.
  """

  metrics: Mapping[str, float] = dataclasses.field(default_factory=dict)

  def metric_map(self) -> Mapping[str, float]:
    """Returns a map of metrics to be exported to MLCompass."""
    return self.metrics


@dataclasses.dataclass
class _WarmupRunResult:
  """Result of the warmup run.

  Attributes:
    num_warmup_steps: The number of warmup steps taken for the cache misses to
      stabilize
    first_step_time_seconds: The time taken for the first step.
    warmup_overhead_seconds: This is the extra time taken to run the benchmark
      to warmup the caches. If it takes n steps for cache misses to stabilize,
      then the warmup overhead is (wall time of n steps) - (wall time of 1 warm
      step) * n. For example, if the cache misses are [100, 120, 130, 130] and
      wall times are [15, 10, 10, 2], then the warmup overhead is (15 + 10 + 10)
      - 2*3 = 29 seconds.
  """

  num_warmup_steps: int = 0
  first_step_time_seconds: float = 0.0
  warmup_overhead_seconds: float = 0.0


@dataclasses.dataclass
class _PostWarmupRunResult:
  """Result of the post warmup run.

  Attributes:
    post_warmup_step_time_seconds: The average step time in seconds after the
      warmup is complete.
    peak_device_memory_mb: The peak device memory usage in MB for a benchmark
      step.
  """

  post_warmup_step_time_seconds: float = 0.0
  peak_device_memory_mb: float = 0.0


class XprofContext:
  """A context manager for Xprof sessions that logs the session URL.

  Attributes:
    name: A name for the xprof session, used in logging.
    enable_xprof: Whether to enable xprof profiling. If False, the context
      manager does nothing.
    session: The xprof_session.XprofSession object if xprof is enabled,
      otherwise None.
    session_id: The ID of the xprof session after it has ended, if xprof was
      enabled. Otherwise None.
  """

  def __init__(self, name: str, enable_xprof: bool):
    self.name = name
    self.enable_xprof = enable_xprof
    self.session = None
    self.session_id = None

  def __enter__(self):
    if self.enable_xprof:
      self.session = xprof_session.XprofSession()
      self.session.start_session(
          host_trace_level=3,
          enable_python_tracer=True,
          host_cpu_profile=True,
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


def get_torch_device(platform: Platform) -> torch.device:
  """Returns the torch device for the given platform."""
  if PLATFORM_DEVICE_MAP[platform] == "tpu":
    return xla_api.tpu_device()
  elif PLATFORM_DEVICE_MAP[platform] == "cuda":
    return torch.device("cuda")
  else:
    raise ValueError(f"Unknown platform: {platform}")


def is_torch_compile(run_mode: RunMode) -> bool:
  """Returns whether the given run mode uses torch.compile."""
  return run_mode == RunMode.COMPILED


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
) -> _WarmupRunResult:
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

  Returns:
    A _WarmupRunResult instance containing the number of warmup steps, the time
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

  with XprofContext("warmup_run", enable_xprof):
    for step in range(MAX_WARMUP_STEPS.value):
      start_time = time.perf_counter()
      out = benchmark_function(model, example_inputs, optimizer)
      if isinstance(out, torch.Tensor):
        device_utils.synchronize(device_name, out)

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

  return _WarmupRunResult(
      num_warmup_steps=num_warmup_steps,
      first_step_time_seconds=timings[0],
      warmup_overhead_seconds=_get_warmup_overhead(timings, num_warmup_steps),
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
    xprof_client: xprof_analysis_client.XprofAnalysisClient | None = None,
) -> _PostWarmupRunResult:
  """Runs the model once after the warmup is complete.

  Args:
    benchmark_function: The benchmark function to run.
    model: The model to run.
    example_inputs: The example inputs to run the model with.
    device: The device the model and input data is on.
    optimizer: The optimizer to use for the model.
    enable_xprof: Whether to enable xprof profiling.
    xprof_client: The xprof client to use for profiling.

  Returns:
    A _PostWarmupRunResult instance containing the average step time and peak
    device memory usage.
  """

  timings = np.zeros(POST_WARMUP_STEPS.value, dtype=np.float64)
  device_utils.reset_peak_memory_stats(_get_device_name(device))
  num_cache_misses = None
  device_name = _get_device_name(device)

  # TODO(bbahl): Calculate the number of post warmup steps based on timing
  # information.
  with XprofContext("post_warmup_run", enable_xprof) as xprof_context:
    for step in range(POST_WARMUP_STEPS.value):
      start_time = time.perf_counter()
      out = benchmark_function(model, example_inputs, optimizer)
      if isinstance(out, torch.Tensor):
        device_utils.synchronize(device_name, out)
      timings[step] = time.perf_counter() - start_time

      # Assert that the cache misses are consistent across steps.
      step_cache_misses = device_utils.cache_miss_count(device_name)
      if num_cache_misses is None:
        num_cache_misses = step_cache_misses
      if step_cache_misses != num_cache_misses:
        raise RuntimeError(
            "Cache misses are not consistent across steps; expected"
            f" {num_cache_misses}, got {step_cache_misses}. This means that the"
            f" model is not fully warmed up after {MAX_WARMUP_STEPS.value}"
            " warmup steps. Consider increasing the number of warmup steps."
        )

  # Calculate the memory usage after the post warmup run is complete. This
  # requires the xprof response for TPU and XLA_CUDA devices, which is only
  # available after the xprof session ends. Memory usage is calculated for TPU
  # and XLA_CUDA devices only when Xprof is enabled.
  memory_usage = device_utils.get_peak_memory_hbm(
      device_name, xprof_context.session_id, xprof_client
  )

  logging.info("Post Warmup Timings: %s", timings)

  return _PostWarmupRunResult(
      post_warmup_step_time_seconds=np.mean(timings),
      peak_device_memory_mb=memory_usage,
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
    xprof_client: xprof_analysis_client.XprofAnalysisClient | None = None,
) -> PerformanceBenchmarkResult:
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

  Returns:
    A PerformanceBenchmarkResult instance containing the results of the
    benchmark.
  """

  if _get_device_name(device) == "cuda" and not torch.cuda.is_available():
    logging.warning("CUDA is not available. Skipping CUDA benchmark.")
    # Return a default result if CUDA is not available to avoid failing
    # presubmits on non-CUDA environments.
    return PerformanceBenchmarkResult()

  _synchronize_all_tensors(example_inputs, device)
  _synchronize_all_tensors(list(model.state_dict().values()), device)
  result_kwargs = {}
  start_time = time.perf_counter()
  result_kwargs |= dataclasses.asdict(
      _warmup_run(
          benchmark_function,
          model,
          example_inputs,
          device,
          optimizer=optimizer,
          enable_xprof=enable_xprof,
      )
  )

  if _do_post_warmup():
    result_kwargs |= dataclasses.asdict(
        _post_warmup_run(
            benchmark_function,
            model,
            example_inputs,
            device,
            optimizer=optimizer,
            enable_xprof=enable_xprof,
            xprof_client=xprof_client,
        )
    )

  return PerformanceBenchmarkResult(
      e2e_wall_time_seconds=time.perf_counter() - start_time,
      **result_kwargs,
  )


def run_quality_benchmark(
    benchmark_model: quality_benchmark_model.QualityBenchmarkModel,
    model_compile: bool,
    dataset_loader: Iterable[Any],
    benchmark_metrics: Sequence[quality_benchmark_model.MetricProducer],
    device: torch.device,
) -> QualityBenchmarkResult:
  """Runs a quality benchmark for a given model.

  Args:
    benchmark_model: The benchmark model object.
    model_compile: Whether to compile the model.
    dataset_loader: A function that loads the dataset.
    benchmark_metrics: A list of metric producers to run.
    device: The device the model and input data is on.

  Returns:
    A QualityBenchmarkResult instance containing the results of the benchmark.
  """
  benchmark_model.initialize()
  start_time = time.perf_counter()
  metric_scores = {}
  for benchmark_metric in benchmark_metrics:
    score = quality_benchmark_model.run_quality_benchmark(
        benchmark_model, model_compile, dataset_loader, benchmark_metric
    )
    # Add metric to score map. Furthermore, it materializes the score.
    metric_scores[benchmark_metric.get_name()] = score.item()
  logging.info("Metric scores (device: %s): %s", device, metric_scores)

  return QualityBenchmarkResult(
      e2e_wall_time_seconds=time.perf_counter() - start_time,
      metrics=metric_scores,
  )
