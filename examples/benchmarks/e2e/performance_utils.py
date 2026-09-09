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
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, TypeAlias, Union

from absl import flags
from absl import logging
from tensorboardX import writer
import torch
from torch_tpu._internal import execution_mode
from torch_tpu._internal import optim
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import common
from examples.benchmarks.e2e import device_utils
from examples.benchmarks.e2e import mlcompass_utils
from examples.benchmarks.e2e import model_utils
from examples.benchmarks.e2e.harness import metrics as metrics_lib
from torch_tpu._internal.benchmarks import xprof_adapter
from tests.distributed import distributed_utils

EagerMode: TypeAlias = execution_mode.EagerMode


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

OPTIM = flags.DEFINE_enum(
    "optim",
    None,
    ["tpu_adamw", "adamw", "adam"],
    "Override the optimizer to use. If not specified, the per-benchmark config"
    " setting will be used.",
)


PATCH_OPTIMIZER_GRAPH_BREAKS = flags.DEFINE_bool(
    "patch_optimizer_graph_breaks",
    False,
    "Whether to patch PyTorch optimizer classes directly to eliminate graph"
    " breaks prior to compilation.",
)

FORCE_FULLGRAPH = flags.DEFINE_bool(
    "force_fullgraph",
    False,
    "Whether to force fullgraph=True for torch.compile across all benchmarks.",
)

DISTRIBUTED_PLATFORMS = (
    common.Platform.GFC_2X2X1,
    common.Platform.GFC_2X2X2,
    common.Platform.GFC_2X2X4,
    common.Platform.GFC_2X4X4,
    common.Platform.B200_4,
)


@functools.lru_cache(maxsize=None)
def get_xprof_client() -> xprof_adapter.XprofAnalysisClient | None:
  """Returns the xprof analysis client.

  This function caches the client to avoid creating multiple connections.
  """
  return xprof_adapter.XprofAnalysisClient()


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
    use_fused_optim: Whether to use fused optimizer.
    optim: The optimizer to use for training ("adamw", "tpu_adamw", "adam").
    fullgraph: Whether to pass fullgraph=True to torch.compile for this
      benchmark.
    is_distributed: Whether to run the benchmark in distributed mode across
      multiple processes/devices. Only required for dual-device single-chip
      platforms (e.g. gfc_1x1x1). Multi-chip platforms (e.g. gfc_2x2x1, b200_4)
      are automatically recognized as distributed.
  """

  supported_platforms: Sequence[common.Platform]
  benchmark_category: benchmark_utils.BenchmarkCategory
  run_mode: common.RunMode
  is_training: bool
  model_and_input_args: ModelAndInputArgs
  model_and_input_factory: Callable[..., Any]
  sync_params: bool = False
  train_factory: Optional[Callable[[], Callable[..., Any]]] = None
  eval_factory: Optional[Callable[[], Callable[..., Any]]] = None
  use_fused_optim: bool = True
  optim: str = "adamw"
  fullgraph: bool = False
  is_distributed: bool = False


# LINT.ThenChange(../../../g3doc/benchmarking.md)


@contextlib.contextmanager
def _run_mode_context(
    run_mode: common.RunMode,
    device: torch.device,
    clear_device_cache: bool = True,
):
  """Context manager to configure the environment for different run modes.

  This includes adjusting environment variables and performing necessary
  cleanups
  like clearing caches and resetting torch.dynamo.

  Args:
    run_mode: The common.RunMode to configure the context for.
    device: The torch device being used.
    clear_device_cache: Whether to clear the device cache.

  Yields:
    None
  """
  original_eager_mode = execution_mode.eager_mode
  new_eager_mode = None

  match run_mode:
    case common.RunMode.EAGER_DEFAULT:
      new_eager_mode = EagerMode.DEFER_NEVER

    case common.RunMode.EAGER_OPTIMIZED:
      new_eager_mode = EagerMode.DEFER_AND_FUSE

    case common.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING:
      new_eager_mode = EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING

    case common.RunMode.COMPILED:
      pass  # Explicitly do nothing

    case _:
      raise ValueError(f"Unexpected run mode: {run_mode}")

  if new_eager_mode is not None:
    execution_mode.eager_mode = new_eager_mode

  try:
    yield
  finally:
    # Change back to the original value.
    execution_mode.eager_mode = original_eager_mode
    # pylint: disable=protected-access
    if clear_device_cache and device.type != "cpu":
      device_utils.clear_cache(device.type)
      if device.type == "tpu":
        tt_testing.reset_eager_state()
    if common.is_torch_compile(run_mode):
      torch._dynamo.reset()


def _setup_absl() -> None:
  """Parses flags from sys.argv if not already parsed and sets up absl logging.

  This function is required for multiprocessing setups. For single processing
  setups absltest.main called in the test file already handles this.
  """
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(sys.argv, known_only=True)
  logging.use_absl_handler()


# Used to make sure that the TensorBoard logging is used only when we run
# one test per bazel run.
_TENSORBOARD_EXPORTED = False


def _export_to_tensorboard(
    result: metrics_lib.MetricsInterface,
    tblog_dir: str,
) -> None:
  """Exports benchmark results to TensorBoard.

  Args:
    result: The benchmark result to export.
    tblog_dir: The directory to write the TensorBoard logs to.

  Raises:
    RuntimeError: If an export has already happened in this process.
  """
  global _TENSORBOARD_EXPORTED
  if _TENSORBOARD_EXPORTED:
    raise RuntimeError(
        "TensorBoard export already happened in this process. "
        "Each run is expected to export only once."
    )

  try:
    os.makedirs(tblog_dir, exist_ok=True)
    tb_writer = writer.SummaryWriter(log_dir=tblog_dir)

    for metric_name, value in result.metric_map().items():
      # Skip metrics that are None (e.g. peak_host_compilation_memory on non-TPU platforms).
      if value is not None:
        tb_writer.add_scalar(metric_name, value, global_step=0)
    tb_writer.close()
  except (OSError, IOError):
    logging.exception("Error writing TensorBoard logs")
  finally:
    _TENSORBOARD_EXPORTED = True


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
  platform = common.PLATFORM.value
  device = common.get_torch_device(platform)
  if hasattr(torch.backends, "tpu"):
    torch.backends.tpu.allow_excess_precision = True
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
  common.seed_rngs()
  weights_dtype = get_torch_dtype(WEIGHTS_DTYPE.value)

  use_torch_compile = common.is_torch_compile(config.run_mode)
  if config.is_training:
    func = config.train_factory()  # pyrefly: ignore[not-callable]
  else:
    func = config.eval_factory()  # pyrefly: ignore[not-callable]
  model_and_input = config.model_and_input_factory(
      model_and_input_args=config.model_and_input_args,
      device=device,
      weights_dtype=weights_dtype,
      is_training=config.is_training,
  )

  # Patch standard PyTorch optimizer classes (e.g. torch.optim.AdamW) to
  # eliminate internal graph breaks (such as state initialization control flow)
  # during torch.compile tracing.
  if PATCH_OPTIMIZER_GRAPH_BREAKS.value:
    optim.patch_optimizer_graph_breaks()

  optim_type = OPTIM.value if OPTIM.value is not None else config.optim
  optimizer = _get_optimizer(
      model_and_input.model,
      is_training=config.is_training,
      use_torch_compile=common.is_torch_compile(config.run_mode),
      use_fused_optim=config.use_fused_optim,
      optim_type=optim_type,
  )

  fullgraph = FORCE_FULLGRAPH.value or config.fullgraph
  if use_torch_compile:
    if config.is_training:
      if not getattr(func, "is_pre_compiled", False):
        func = device_utils.torch_compile(
            func,
            device.type,
            dynamic=BOUNDED_DYNAMIC.value,
            fullgraph=fullgraph,
        )
    else:
      model_and_input.model = device_utils.torch_compile(
          model_and_input.model,
          device.type,
          dynamic=BOUNDED_DYNAMIC.value,
          fullgraph=fullgraph,
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
          "  average_post_warmup_device_time (seconds): %s\n"
          "  peak_device_memory (MB): %s\n"
          "  e2e_wall_time (seconds): %s\n"
          "  warmup_session_xprof_url: %s\n"
          "  post_warmup_run_session_xprof_url: %s\n"
          "  peak_host_compilation_memory (MB): %s\n",
          test_method_name,
          benchmark_name,
          microbenchmark_name,
          config.is_training,
          common.PLATFORM.value,
          WEIGHTS_DTYPE.value,
          rank,
          world_size,
          result.num_warmup_steps,
          result.first_step_time_seconds,
          result.warmup_overhead_seconds,
          result.post_warmup_step_time_seconds,
          result.average_post_warmup_device_time_seconds,
          result.peak_device_memory_mb,
          result.e2e_wall_time_seconds,
          result.warmup_session_xprof_url,
          result.post_warmup_run_session_xprof_url,
          result.peak_host_compilation_memory_mb,
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
      _export_to_tensorboard(result, tblog_dir)

  if torch.distributed.is_initialized():
    logging.info("Process %s reaching final barrier...", rank)
    torch.distributed.barrier()
    logging.info("Process %s passed final barrier.", rank)


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
  platform = common.PLATFORM.value
  if platform == common.Platform.B200_4:
    # A single B200 device is roughly equivalent to two GFC devices,
    # so double the batch size on B200 to make a fairer comparison.
    config.model_and_input_args.batch_size = (
        config.model_and_input_args.batch_size * 2  # pyrefly: ignore[unsupported-operation]
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
  elif platform == common.Platform.GFC_1X1X1:
    singlehost_wrapper.prepare_tpu_environment(world_size=2)
    distributed_utils.dist_run(
        2,
        run_torch_tpu_task,
        run_single_process_benchmark,
        (
            config,
            test_method_name,
            benchmark_name,
            microbenchmark_name,
        ),
    )
  elif platform == common.Platform.GFC_2X2X1:
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
    raise ValueError(f"No worker function for platform: {platform}")


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

  platform = common.PLATFORM.value
  if config.is_distributed or platform in DISTRIBUTED_PLATFORMS:
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
    model: torch.nn.Module,
    *,
    is_training: bool,
    use_torch_compile: bool,
    use_fused_optim: bool = True,
    optim_type: str = "adamw",
) -> torch.optim.Optimizer | None:
  """Returns optimizer for training, or None for inference."""
  # Model parameters can be empty for some layers, e.g. Dropout.
  if not is_training or not list(model.parameters()):
    return None

  if optim_type == "tpu_adamw":
    optimizer = optim.AdamW
  elif optim_type == "adamw":
    optimizer = torch.optim.AdamW
  elif optim_type == "adam":
    optimizer = torch.optim.Adam
  else:
    raise ValueError(f"Unsupported optimizer type: {optim_type}")

  return optimizer(
      model.parameters(),
      lr=0.1,  # Gigantic LR for testing.
      capturable=use_torch_compile,
      fused=use_fused_optim,
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
