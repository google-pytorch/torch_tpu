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

"""Utilities for running end-to-end performance benchmarks."""

import contextlib
import dataclasses
import functools
import gc
import os
import sys
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

from absl import flags
from absl import logging
from tensorboardX import writer
import torch
from torch_tpu._internal import execution_mode
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import device_utils
from torch_tpu._internal.utils import log_utils
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import mlcompass_utils
from examples.benchmarks.e2e import model_utils
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.xprof import xprof_analysis_client

log_utils.log_to_stderr()


WEIGHTS_DTYPE = flags.DEFINE_string(
    "weights_dtype",
    "bfloat16",
    "The dtype to use for model weights.",
)

ENABLE_XPROF = flags.DEFINE_bool(
    "enable_xprof",
    False,
    "Whether to enable xprof profiling for post warmup run.",
)

BOUNDED_DYNAMIC = flags.DEFINE_bool(
    "bounded_dynamic",
    False,
    "Whether to run the E2E benchmarks with bounded dynamic shapes.",
)

DISTRIBUTED_PLATFORMS = (
    benchmark_utils.Platform.GFC_2X2X1,
    benchmark_utils.Platform.GFC_2X2X2,
    benchmark_utils.Platform.GFC_2X2X4,
    benchmark_utils.Platform.GFC_2X4X4,
    benchmark_utils.Platform.B200_4,
)


@functools.lru_cache(maxsize=None)
def get_xprof_client() -> xprof_analysis_client.XprofAnalysisClient | None:
  """Returns the xprof analysis client.

  This function caches the client to avoid creating multiple connections.
  """
  return xprof_analysis_client.XprofAnalysisClient()


@dataclasses.dataclass
class ModelAndInputArgs:
  """Args to get the model and example inputs.

  Attributes:
    model_name: The name of the model to benchmark.
    sequence_length: The sequence length of the input.
    batch_size: The batch size of the input.
    is_bounded_dynamic: Whether to run the E2E benchmarks with bounded dynamic
      shapes.
    custom_kwargs: Custom kwargs that might be needed to get the model and
      example inputs.
  """

  model_name: Optional[str] = None
  sequence_length: Optional[Union[int, model_utils.DynamicDimension]] = None
  batch_size: Optional[Union[int, model_utils.DynamicDimension]] = None
  is_bounded_dynamic: bool = False
  custom_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)


# LINT.IfChange
@dataclasses.dataclass
class PerformanceBenchmarkConfig:
  """The config for a performance benchmark.

  Attributes:
    supported_platforms: The platforms the benchmark supports. If the current
      platform is not in this sequence, the test will be skipped.
    benchmark_category: The category of the benchmark. This defines how to get
      the model and example inputs and the benchmark function to run.
    run_mode: The mode to run the benchmark in. This is used to set environment
      variables.
    is_training: Whether the benchmark is for training. If True, the benchmark
      will use the training mode of the model and will run the optimizer.
    model_and_input_args: The args to get the model and example inputs.
    model_and_input_factory: Factory to create the model and inputs.
    sync_params: Whether to eagerly synchronize parameter gradients inside
      timing loops.
    train_factory: Optional factory to create the training benchmark function.
    eval_factory: Optional factory to create the inference benchmark function.
  """

  supported_platforms: Sequence[benchmark_utils.Platform]
  benchmark_category: benchmark_utils.BenchmarkCategory
  run_mode: benchmark_utils.RunMode
  is_training: bool
  model_and_input_args: ModelAndInputArgs
  model_and_input_factory: Callable[..., Any]
  sync_params: bool = False
  train_factory: Optional[Callable[[], Callable[..., Any]]] = None
  eval_factory: Optional[Callable[[], Callable[..., Any]]] = None


# LINT.ThenChange(../../../g3doc/benchmarking.md)


@contextlib.contextmanager
def _run_mode_context(run_mode: benchmark_utils.RunMode, device: torch.device):
  """Context manager to configure the environment for different run modes.

  This includes adjusting environment variables and performing necessary
  cleanups
  like clearing caches and resetting torch.dynamo.

  Args:
    run_mode: The benchmark_utils.RunMode to configure the context for.
    device: The torch device being used.

  Yields:
    None
  """
  original_eager_mode = execution_mode.get_eager_mode()
  new_eager_mode = None

  match run_mode:
    case benchmark_utils.RunMode.EAGER_DEFAULT:
      new_eager_mode = execution_mode.EagerMode.DEFER_NEVER

    case benchmark_utils.RunMode.EAGER_OPTIMIZED:
      new_eager_mode = execution_mode.EagerMode.DEFER_AND_FUSE

    case benchmark_utils.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING:
      new_eager_mode = execution_mode.EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING

    case benchmark_utils.RunMode.COMPILED:
      pass  # Explicitly do nothing

    case _:
      raise ValueError(f"Unexpected run mode: {run_mode}")

  if new_eager_mode is not None:
    execution_mode.set_eager_mode(new_eager_mode)

  try:
    yield
  finally:
    # Change back to the original value.
    execution_mode.set_eager_mode(original_eager_mode)
    # pylint: disable=protected-access
    device_utils.clear_cache(device.type)
    if benchmark_utils.is_torch_compile(run_mode):
      torch._dynamo.reset()


def _setup_absl() -> None:
  """Parses flags from sys.argv if not already parsed and sets up absl logging.

  This function is required for multiprocessing setups. For single processing
  setups absltest.main called in the test file already handles this.
  """
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(sys.argv, known_only=True)
  logging.use_absl_handler()


# Not using keyword arguments here because the google version of torchrun
# doesn't support passing keyword arguments to the worker function.
def run_single_process_benchmark(
    config: PerformanceBenchmarkConfig,
    test_method_name: str,
    benchmark_name: str,
    microbenchmark_name: str | None = None,
) -> None:
  """Runs the benchmark for the given config.

  Args:
    config: The benchmark config. Contains the arguments to run the benchmark.
    test_method_name: The name of the test method. This is used for logging the
      benchmark results.
    benchmark_name: The name of the benchmark. This combined with
      test_method_name is used to uniquely identify the benchmark in MLCompass.
    microbenchmark_name: This is used to export microbenchmark results to
      MLCompass when a benchmark test method is composed of multiple
      microbenchmarks. See go/mlcompass-microbenchmark-guide for more details.
  """
  gc.collect()

  _setup_absl()
  rank = int(os.environ.get("RANK", "0"))
  logging.info("Process %s starting run_single_process_benchmark", rank)
  world_size = int(os.environ.get("WORLD_SIZE", "1"))
  platform = benchmark_utils.PLATFORM.value
  device = benchmark_utils.get_torch_device(platform)
  # Seed random number generators for reproducibility. This should be done after
  # initializing the device.
  if config.is_training:
    assert (
        config.train_factory is not None
    ), "train_factory must be provided for training."
  else:
    assert (
        config.eval_factory is not None
    ), "eval_factory must be provided for inference."

  # Seed random number generators for reproducibility. This should be done after
  # initializing the device.
  benchmark_utils.seed_rngs()
  weights_dtype = get_torch_dtype(WEIGHTS_DTYPE.value)

  use_torch_compile = benchmark_utils.is_torch_compile(config.run_mode)
  if config.is_training:
    func = config.train_factory()
  else:
    func = config.eval_factory()
  model_and_input = config.model_and_input_factory(
      model_and_input_args=config.model_and_input_args,
      device=device,
      weights_dtype=weights_dtype,
      is_training=config.is_training,
  )

  if use_torch_compile:
    if config.is_training:
      func = device_utils.torch_compile(
          func, device.type, dynamic=BOUNDED_DYNAMIC.value
      )
    else:
      model_and_input.model = device_utils.torch_compile(
          model_and_input.model, device.type, dynamic=BOUNDED_DYNAMIC.value
      )
  optimizer = _get_optimizer(
      model_and_input.model,
      is_training=config.is_training,
      use_torch_compile=benchmark_utils.is_torch_compile(config.run_mode),
  )
  # Only enable xprof for rank 0 process.
  enable_xprof = ENABLE_XPROF.value and rank == 0

  xprof_client = None
  if enable_xprof:
    xprof_client = get_xprof_client()

  benchmark_succeeded = True
  result = None
  benchmark_exception = None
  try:
    with _run_mode_context(config.run_mode, device):
      result = benchmark_utils.run_performance_benchmark(
          func,
          model_and_input.model,
          model_and_input.example_inputs,
          device,
          enable_xprof=enable_xprof,
          optimizer=optimizer,
          xprof_client=xprof_client,
          sync_params=config.sync_params,
          is_bounded_dynamic=BOUNDED_DYNAMIC.value,
      )
      logging.info(
          "Performance Benchmark Results:\n"
          "  Test: %s\n"
          "  benchmark: %s\n"
          "  microbenchmark: %s\n"
          "  is_training: %s\n"
          "  platform: %s\n"
          "  weights_dtype: %s\n"
          "  rank(0-indexed): %s\n"
          "  world_size: %s\n"
          "  num_warmup_steps: %s\n"
          "  first_step_time (seconds): %s\n"
          "  warmup_overhead (seconds): %s\n"
          "  average_step_time (seconds): %s\n"
          "  peak_device_memory (MB): %s\n"
          "  e2e_wall_time (seconds): %s\n"
          "  warmup_session_xprof_url: %s\n"
          "  post_warmup_run_session_xprof_url: %s",
          test_method_name,
          benchmark_name,
          microbenchmark_name,
          config.is_training,
          benchmark_utils.PLATFORM.value,
          WEIGHTS_DTYPE.value,
          rank,
          world_size,
          result.num_warmup_steps,
          result.first_step_time_seconds,
          result.warmup_overhead_seconds,
          result.post_warmup_step_time_seconds,
          result.peak_device_memory_mb,
          result.e2e_wall_time_seconds,
          result.warmup_session_xprof_url,
          result.post_warmup_run_session_xprof_url,
      )
  except Exception as e:
    logging.exception("Benchmark failed: %s", e)
    benchmark_succeeded = False
    benchmark_exception = e

  # Only export results from the rank 0 process to avoid duplicate entries in
  # MLCompass.
  if benchmark_utils.MLCOMPASS_TRACKING_ID.value and rank == 0:
    mlcompass_utils.export_to_mlcompass(
        platform,
        result,
        benchmark_utils.BASE_CL.value,
        benchmark_utils.MLCOMPASS_TRACKING_ID.value,
        benchmark_utils.MLCOMPASS_EXECUTION_MODE.value,
        test_method_name=test_method_name,
        benchmark_name=benchmark_name,
        microbenchmark_name=microbenchmark_name,
        succeeded=benchmark_succeeded,
        pending_cl=benchmark_utils.PENDING_CL.value,
        benchmark_group=benchmark_utils.BENCHMARK_GROUP.value,
    )

  if not benchmark_succeeded and benchmark_exception is not None:
    raise benchmark_exception

  if (
      benchmark_succeeded
      and result is not None
      and benchmark_utils.ENABLE_TENSORBOARD_LOGGING.value
      and rank == 0
  ):
    tblog_dir = os.environ.get(benchmark_utils.TENSORBOARD_OUTPUT_ENV_VAR.value)
    if tblog_dir:
      try:
        tb_writer = writer.SummaryWriter(log_dir=tblog_dir)
        metric_tag = (
            f"{mlcompass_utils.TEAM_NAME}/{platform.value}/{test_method_name}/"
            f"{benchmark_name}"
        )
        if microbenchmark_name:
          metric_tag = f"{metric_tag}/{microbenchmark_name}"

        for metric_name, value in result.metric_map().items():
          tb_writer.add_scalar(
              f"{metric_tag}/{metric_name}", value, global_step=0
          )
        tb_writer.close()
      except (OSError, IOError):
        logging.exception("Error writing TensorBoard logs")


def run_torch_tpu_task(
    worker_func: Callable[..., Any],
    extra_worker_func_args: Tuple[Any, ...],
) -> None:
  rank = os.environ.get("RANK", "0")
  logging.info("Process %s initializing process group...", rank)
  torch.distributed.init_process_group(backend="tpu_dist")
  logging.info("Process %s process group initialized.", rank)
  worker_func(*extra_worker_func_args)
  torch.distributed.destroy_process_group()


def _run_cuda_task(
    worker_func: Callable[..., Any],
    extra_worker_func_args: Tuple[Any, ...],
) -> None:
  rank = int(os.environ.get("RANK", "0"))
  world_size = int(os.environ.get("WORLD_SIZE", "1"))
  if not torch.cuda.is_available():
    raise RuntimeError(f"CUDA is not available on rank {rank}.")

  torch.distributed.init_process_group(
      backend="nccl", rank=rank, world_size=world_size
  )
  torch.cuda.set_device(rank)

  worker_func(*extra_worker_func_args)
  torch.distributed.destroy_process_group()


def _run_distributed_benchmark(
    config: PerformanceBenchmarkConfig,
    *,
    test_method_name: str,
    benchmark_name: str,
    microbenchmark_name: str | None = None,
) -> None:
  """Runs the benchmark for the given config."""
  platform = benchmark_utils.PLATFORM.value
  if platform == benchmark_utils.Platform.B200_4:
    # A single B200 device is roughly equivalent to two GFC devices,
    # so double the batch size on B200 to make a fairer comparison.
    config.model_and_input_args.batch_size = (
        config.model_and_input_args.batch_size * 2
    )
    distributed_utils.dist_run(
        4,
        _run_cuda_task,
        run_single_process_benchmark,
        (
            config,
            test_method_name,
            benchmark_name,
            microbenchmark_name,
        ),
    )
  elif platform == benchmark_utils.Platform.GFC_2X2X1:
    singlehost_wrapper.prepare_tpu_environment(world_size=8)
    distributed_utils.dist_run(
        8,
        run_torch_tpu_task,
        run_single_process_benchmark,
        (
            config,
            test_method_name,
            benchmark_name,
            microbenchmark_name,
        ),
    )
  else:
    raise ValueError(
        f"No worker function for platform: {benchmark_utils.PLATFORM.value}"
    )


def run_benchmark(
    config: PerformanceBenchmarkConfig,
    *,
    test_method_name: str,
    benchmark_name: str,
    microbenchmark_name: str | None = None,
) -> None:
  """Runs the benchmark for the given config.

  Args:
    config: The benchmark config. Contains the arguments to run the benchmark.
    test_method_name: The name of the test method. This is used for logging the
      benchmark results.
    benchmark_name: The name of the benchmark. This combined with
      test_method_name is used to uniquely identify the benchmark in MLCompass.
    microbenchmark_name: This is used to export microbenchmark results to
      MLCompass when a benchmark test method is composed of multiple
      microbenchmarks. See go/mlcompass-microbenchmark-guide for more details.
  """

  platform = benchmark_utils.PLATFORM.value
  if platform in DISTRIBUTED_PLATFORMS:
    logging.info("Running distributed benchmark on platform %s", platform)
    _run_distributed_benchmark(
        config,
        test_method_name=test_method_name,
        benchmark_name=benchmark_name,
        microbenchmark_name=microbenchmark_name,
    )
  else:
    logging.info("Running single process benchmark on platform %s", platform)
    run_single_process_benchmark(
        config,
        test_method_name=test_method_name,
        benchmark_name=benchmark_name,
        microbenchmark_name=microbenchmark_name,
    )


def _get_optimizer(
    model: torch.nn.Module, *, is_training: bool, use_torch_compile: bool
) -> torch.optim.Optimizer | None:
  """Returns optimizer for training, or None for inference."""
  if not is_training:
    return None
  return torch.optim.AdamW(
      model.parameters(),
      lr=0.1,  # Gigantic LR for testing.
      capturable=use_torch_compile,
      # The non-fused version of adam will expand foreach into loops and
      # increase the graph size. The compile time increase is especially
      # obvious when compiling AdamW with torch.compile.
      # We should consider implementing 'aten::_fused_adamw_' to both
      # improve compile time and increase performance.
      fused=False,
  )


def get_torch_dtype(dtype_str: str) -> torch.dtype:
  if dtype_str == "bfloat16":
    return torch.bfloat16
  elif dtype_str == "float16":
    return torch.float16
  elif dtype_str == "float32":
    return torch.float32
  else:
    raise ValueError(f"Unknown model dtype: {dtype_str}")
