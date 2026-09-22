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

"""Standalone forward-pass benchmark for the Muse Glimmer 30B example.

Self-contained: device selection, timing, and metric reporting are implemented
here so that the example runs unmodified on TPU, GPU (B200) and CPU, both as a
plain Python script and as a test target.

Metrics are printed to stdout, and additionally written as TensorBoard scalars
when `TENSORBOARD_OUTPUT_DIR` is set (this is how the OSS benchmark runners
collect results). The scalar names match the ones used by the e2e benchmark
harness, so the existing `benchmark_registry.pbtxt` metric patterns (e.g.
`.*/post_warmup_step_time_ms$`) apply unchanged.
"""

import dataclasses
import os
import time
from typing import Mapping

from absl import flags
from absl import logging
import torch
from examples.muse_glimmer import model as model_lib

_DEVICE = flags.DEFINE_string(
    "device",
    "auto",
    "Device to run on: 'auto', 'tpu', 'cuda' or 'cpu'.",
)
_DTYPE = flags.DEFINE_enum(
    "dtype",
    "bfloat16",
    ["bfloat16", "float32"],
    "Default dtype of the model weights and activations.",
)
_COMPILE = flags.DEFINE_bool(
    "compile",
    None,
    "Whether to run the model under torch.compile. Defaults to true on"
    " accelerators and false on CPU, where the inductor backend needs a host"
    " C++ compiler that is not always available.",
)

_NUM_LAYERS = flags.DEFINE_integer(
    "num_layers", None, "Overrides the number of decoder layers."
)
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size", None, "Overrides the (global) batch size."
)
_SEQ_LEN = flags.DEFINE_integer(
    "seq_len", None, "Overrides the sequence length."
)
_VOCAB_SIZE = flags.DEFINE_integer(
    "vocab_size", None, "Overrides the vocabulary size."
)
_WARMUP_STEPS = flags.DEFINE_integer(
    "warmup_steps", 3, "Number of warmup steps (the first one compiles)."
)
_MEASURE_STEPS = flags.DEFINE_integer(
    "measure_steps", 5, "Number of measured steps after warmup."
)
_SMOKE_TEST = flags.DEFINE_bool(
    "smoke_test",
    None,
    "Run a tiny configuration instead of the full 30B model. Defaults to true"
    " when no accelerator is available.",
)
_BENCHMARK_NAME = flags.DEFINE_string(
    "benchmark_name",
    "muse_glimmer_30b_forward",
    "Name used when reporting metrics.",
)

_BYTES_IN_MB = 2**20

# Environment variables shared with the e2e benchmark harness/runners.
_TENSORBOARD_OUTPUT_DIR_ENV_VAR = "TENSORBOARD_OUTPUT_DIR"
_PLATFORM_ENV_VAR = "BENCHMARK_PLATFORM"

_TEAM_NAME = "torch_tpu"
_FRAMEWORK = "torch"

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Small configuration used for smoke tests (CPU, presubmit, local runs).
_SMOKE_CONFIG = model_lib.Config(
    vocab_size=8192,
    num_layers=2,
    dmodel=512,
    dff=1536,
    dhead=64,
    num_q_heads=8,
    num_kv_heads=2,
    swa_size=128,
    batch_size=1,
    seq_len=512,
)


@dataclasses.dataclass
class Metrics:
  """Timing and memory results of a benchmark run.

  Attributes:
    num_warmup_steps: Number of warmup steps that were run.
    first_step_time_seconds: Wall time of the first (compiling) step.
    warmup_overhead_seconds: Extra wall time spent in warmup compared to running
      the same number of steps at the measured warm step time.
    post_warmup_step_time_seconds: Average wall time of a measured step.
    peak_device_memory_mb: Peak device memory during the measured steps, or None
      if the device does not report it.
    e2e_wall_time_seconds: Wall time of the whole benchmark, including model
      construction.
  """

  num_warmup_steps: int = 0
  first_step_time_seconds: float = 0.0
  warmup_overhead_seconds: float = 0.0
  post_warmup_step_time_seconds: float = 0.0
  peak_device_memory_mb: float | None = None
  e2e_wall_time_seconds: float = 0.0

  def metric_map(self) -> Mapping[str, float]:
    """Returns the metrics keyed by their reported names."""
    metrics = {
        "num_warmup_steps": float(self.num_warmup_steps),
        "first_step_time_seconds": self.first_step_time_seconds,
        "first_step_time_ms": self.first_step_time_seconds * 1000.0,
        "warmup_overhead_seconds": self.warmup_overhead_seconds,
        "warmup_overhead_ms": self.warmup_overhead_seconds * 1000.0,
        "post_warmup_step_time_seconds": self.post_warmup_step_time_seconds,
        "post_warmup_step_time_ms": self.post_warmup_step_time_seconds * 1000.0,
    }
    if self.peak_device_memory_mb is not None:
      metrics["peak_device_memory_mb"] = self.peak_device_memory_mb
    return metrics


def resolve_device(name: str = "auto") -> torch.device:
  """Resolves the device to run on, preferring the attached accelerator."""
  if name != "auto":
    return torch.device(name)

  accelerator = None
  try:
    accelerator = torch.accelerator.current_accelerator()
  except (AttributeError, RuntimeError):
    accelerator = None

  if accelerator is not None and accelerator.type != "cpu":
    try:
      available = torch.accelerator.device_count() > 0
    except (AttributeError, RuntimeError):
      available = False
    if available:
      return torch.device(accelerator.type)

  if torch.cuda.is_available():
    return torch.device("cuda")
  return torch.device("cpu")


def synchronize(device: torch.device) -> None:
  """Blocks until all work queued on `device` has completed."""
  if device.type == "cpu":
    return
  torch.accelerator.synchronize()


def reset_peak_memory(device: torch.device) -> None:
  """Resets the peak memory counter of `device`, if it has one."""
  if device.type == "cpu":
    return
  try:
    torch.accelerator.reset_peak_memory_stats(device)
  except (AttributeError, RuntimeError, TypeError):
    logging.info("Peak memory stats are not supported on %s.", device)


def peak_memory_mb(device: torch.device) -> float | None:
  """Returns the peak memory of `device` in MB, or None if unsupported."""
  if device.type == "cpu":
    return None
  try:
    return torch.accelerator.max_memory_allocated(device) / _BYTES_IN_MB
  except (AttributeError, RuntimeError, TypeError):
    return None


def compile_backend(device: torch.device) -> str:
  """Returns the torch.compile backend to use on `device`."""
  if device.type == "tpu":
    return "tpu"

  # pylint: disable=protected-access
  # Inductor spawns compile workers lazily; doing so during the timed steps
  # skews the measurements, so compile in-process instead.
  torch._inductor.config.compile_threads = 1
  # Inductor generates C++ host code and needs a compiler to build it.
  if getattr(torch._inductor.config.cpp, "cxx", (None,))[0] is None:
    for cxx in ("/usr/bin/clang++", "/usr/bin/g++"):
      if os.path.exists(cxx):
        torch._inductor.config.cpp.cxx = (None, cxx)
        break
  # pylint: enable=protected-access
  return "inductor"


def make_config(device: torch.device) -> model_lib.Config:
  """Builds the model config from the flags, for the given device."""
  smoke_test = _SMOKE_TEST.value
  if smoke_test is None:
    smoke_test = device.type == "cpu"

  config = (
      dataclasses.replace(_SMOKE_CONFIG) if smoke_test else model_lib.Config()
  )

  if _NUM_LAYERS.value is not None:
    config.num_layers = _NUM_LAYERS.value
  if _BATCH_SIZE.value is not None:
    config.batch_size = _BATCH_SIZE.value
  if _SEQ_LEN.value is not None:
    config.seq_len = _SEQ_LEN.value
  if _VOCAB_SIZE.value is not None:
    config.vocab_size = _VOCAB_SIZE.value
  return config


def build_model(
    config: model_lib.Config, device: torch.device, dtype: torch.dtype
) -> model_lib.MuseGlimmer30B:
  """Builds and initializes the model directly on `device`."""
  torch.manual_seed(42)
  previous_dtype = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  try:
    with torch.device(device):
      m = model_lib.MuseGlimmer30B(config)
    model_lib.init_nn_module(m)
  finally:
    torch.set_default_dtype(previous_dtype)

  m.eval()
  synchronize(device)
  return m


def make_inputs(config: model_lib.Config, device: torch.device) -> torch.Tensor:
  """Creates a random token batch on `device`."""
  inputs = torch.randint(
      0,
      config.vocab_size,
      (config.batch_size, config.seq_len),
      device=device,
  )
  synchronize(device)
  return inputs


def measure_forward(
    m: model_lib.MuseGlimmer30B,
    inputs: torch.Tensor,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
) -> Metrics:
  """Times `warmup_steps` warmup steps followed by `measure_steps` steps."""
  if warmup_steps < 1:
    raise ValueError("warmup_steps must be at least 1")
  if measure_steps < 1:
    raise ValueError("measure_steps must be at least 1")

  @torch.no_grad()
  def run_step() -> None:
    m(inputs)
    synchronize(device)

  warmup_times = []
  for step in range(warmup_steps):
    start = time.perf_counter()
    run_step()
    warmup_times.append(time.perf_counter() - start)
    logging.info(
        "Warmup step %d/%d: %.3f ms",
        step + 1,
        warmup_steps,
        warmup_times[-1] * 1000.0,
    )

  reset_peak_memory(device)

  start = time.perf_counter()
  for step in range(measure_steps):
    step_start = time.perf_counter()
    run_step()
    logging.info(
        "Measured step %d/%d: %.3f ms",
        step + 1,
        measure_steps,
        (time.perf_counter() - step_start) * 1000.0,
    )
  step_time = (time.perf_counter() - start) / measure_steps

  return Metrics(
      num_warmup_steps=warmup_steps,
      first_step_time_seconds=warmup_times[0],
      warmup_overhead_seconds=sum(warmup_times) - step_time * warmup_steps,
      post_warmup_step_time_seconds=step_time,
      peak_device_memory_mb=peak_memory_mb(device),
  )


def export_metrics(
    name: str, metrics: Metrics, run_mode: str, succeeded: bool = True
) -> None:
  """Writes the metrics as TensorBoard scalars, if an output dir is set."""
  log_dir = os.environ.get(  # PY_ENVIRON_OK=test
      _TENSORBOARD_OUTPUT_DIR_ENV_VAR, None
  )
  if not log_dir:
    return

  platform = os.environ.get(_PLATFORM_ENV_VAR, "cpu")  # PY_ENVIRON_OK=test
  prefix = f"{_TEAM_NAME}/{platform}/{name}/{run_mode}/{_FRAMEWORK}"

  try:
    from tensorboardX import writer  # pylint: disable=g-import-not-at-top
  except ImportError:
    logging.warning("tensorboardX is unavailable; skipping metric export.")
    return

  try:
    os.makedirs(log_dir, exist_ok=True)
    with writer.SummaryWriter(log_dir=log_dir) as tb_writer:
      if succeeded:
        for metric_name, value in metrics.metric_map().items():
          tb_writer.add_scalar(f"{prefix}/{metric_name}", value, global_step=0)
        tb_writer.add_scalar(
            f"{prefix}/wall_time", metrics.e2e_wall_time_seconds, global_step=0
        )
      tb_writer.add_scalar(f"{prefix}/succeeded", int(succeeded), global_step=0)
  except OSError:
    logging.exception("Error writing TensorBoard logs")


def log_metrics(name: str, config: model_lib.Config, metrics: Metrics) -> None:
  """Logs a human-readable summary of the run."""
  lines = [f"{name}:", f"  config: {config}"]
  for metric_name, value in metrics.metric_map().items():
    lines.append(f"  {metric_name}: {value:.3f}")
  lines.append(f"  e2e_wall_time_seconds: {metrics.e2e_wall_time_seconds:.3f}")
  print("\n".join(lines), flush=True)


def run_benchmark(
    config: model_lib.Config | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    use_compile: bool | None = None,
    warmup_steps: int | None = None,
    measure_steps: int | None = None,
    name: str | None = None,
) -> Metrics:
  """Runs the forward benchmark; unset arguments fall back to the flags.

  Args:
    config: Model configuration.
    device: Device to run on.
    dtype: Default dtype of weights and activations.
    use_compile: Whether to run under torch.compile.
    warmup_steps: Number of warmup steps.
    measure_steps: Number of measured steps.
    name: Benchmark name used when reporting metrics.

  Returns:
    The metrics of the run.
  """
  start = time.perf_counter()

  device = device if device is not None else resolve_device(_DEVICE.value)
  config = config if config is not None else make_config(device)
  dtype = dtype if dtype is not None else _DTYPES[_DTYPE.value]
  if use_compile is None:
    use_compile = (
        _COMPILE.value if _COMPILE.value is not None else device.type != "cpu"
    )

  warmup_steps = (
      warmup_steps if warmup_steps is not None else _WARMUP_STEPS.value
  )
  measure_steps = (
      measure_steps if measure_steps is not None else _MEASURE_STEPS.value
  )
  name = name if name is not None else _BENCHMARK_NAME.value

  logging.info("Running %s on %s with %s", name, device, config)

  m = build_model(config, device, dtype)
  if use_compile:
    m.apply_torch_compile(backend=compile_backend(device))
  logging.info("Model parameters: %d", model_lib.count_parameters(m))

  inputs = make_inputs(config, device)
  metrics = measure_forward(m, inputs, device, warmup_steps, measure_steps)
  metrics.e2e_wall_time_seconds = time.perf_counter() - start

  log_metrics(name, config, metrics)
  export_metrics(name, metrics, run_mode="compiled" if use_compile else "eager")
  return metrics
