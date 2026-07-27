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

"""Performance utilities for TorchAx benchmarks."""

import enum
import os
import time
from typing import Any

from absl import logging
import jax
import numpy as np
import optax
import torch
from torch_tpu._internal.utils import log_utils
from examples.benchmarks.e2e import benchmark_utils as pt_benchmark_utils
from examples.benchmarks.e2e import common as pt_common
from examples.benchmarks.e2e import device_utils
from examples.benchmarks.e2e.harness import metrics as metrics_lib
from examples.benchmarks.e2e import mlcompass_utils
from examples.benchmarks.e2e import performance_utils as pt_performance_utils
import torchax
from torchax import interop  # pylint: disable=unused-import  # noqa: F401
from torchax import train  # pylint: disable=unused-import  # noqa: F401

from torch_tpu._internal.shims.xprof import traceme

# Monkeypatch torchax.tensor.Environment._to_copy to handle raw Python scalars (int, float, bool)
# passed during functorch/vmap tracing, converting them to PyTorch tensors on the fly.
_original_to_copy = torchax.tensor.Environment._to_copy


def _patched_to_copy(self, the_tensor, new_dtype, new_device):
  if isinstance(the_tensor, (int, float, bool)):
    the_tensor = torch.tensor(the_tensor)
  return _original_to_copy(self, the_tensor, new_dtype, new_device)


torchax.tensor.Environment._to_copy = _patched_to_copy

log_utils.log_to_stderr()


class ModelBenchmarkOutputType(enum.Enum):
  """Enum for the type of return in ModelBenchmarkOutput."""

  LOSS = "loss"
  SAMPLE = "sample"
  LOGITS = "logits"
  RAW = "raw"
  ALL = "all"


@jax.tree_util.register_pytree_node_class
class ModelBenchmarkOutput:
  """A structure to hold the type of return and the actual data."""

  def __init__(self, return_type: ModelBenchmarkOutputType, data: Any):
    self.return_type = return_type
    self.data = data

  def tree_flatten(self):
    """Flattens the object into children (JAX arrays) and auxiliary data (metadata)."""
    return (self.data,), (self.return_type,)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """Reconstructs the object from auxiliary data and children."""
    return cls(aux_data[0], children[0])


def _sync_jax_device(x):
  """Synchronizes JAX device by blocking until ready."""

  def sync_leaf(leaf):
    if hasattr(leaf, "jax"):
      leaf.jax().block_until_ready()
    elif hasattr(leaf, "block_until_ready"):
      leaf.block_until_ready()
    return leaf

  jax.tree_util.tree_map(sync_leaf, x)


def _call_functional_model(model_jittable, params, buffers, inputs):
  """Calls the jittable module functional_call with inputs.

  Args:
    model_jittable: The JittableModule instance.
    params: The parameters for the functional call.
    buffers: The buffers for the functional call.
    inputs: The inputs to the model. This can be a dictionary of keyword
      arguments (e.g., for Wan models) or a single positional tensor (e.g., for
      ResNet). We construct args and kwargs dynamically to avoid duplicating the
      functional_call line.

  Returns:
    The output of the functional call.
  """
  if isinstance(inputs, dict):
    args = ()
    kwargs = inputs
  elif isinstance(inputs, tuple):
    args = inputs
    kwargs = {}
  else:
    args = (inputs,)
    kwargs = {}

  # Call the functional model and get the result. The functional_call is used
  # to represent the model as a function that can be called with JAX-compatible
  # arguments, which is required for JAX JIT compilation.
  res = model_jittable.functional_call(
      "forward", params, buffers, *args, **kwargs
  )

  # Prioritize explicit precomputed loss if available.
  if hasattr(res, ModelBenchmarkOutputType.LOSS.value) and res.loss is not None:
    return ModelBenchmarkOutput(ModelBenchmarkOutputType.LOSS, res.loss)

  # Otherwise, collect all available tensor outputs.
  collected_tensors = {}

  # Check common explicit attributes first
  for attr_name in (
      ModelBenchmarkOutputType.SAMPLE.value,
      ModelBenchmarkOutputType.LOGITS.value,
  ):
    if hasattr(res, attr_name):
      val = getattr(res, attr_name)
      if isinstance(val, torch.Tensor):
        collected_tensors[attr_name] = val

  # Next, iterate via items() (common for HF dict-like dataclasses)
  if hasattr(res, "items"):
    for k, v in res.items():
      if isinstance(v, torch.Tensor) and k not in collected_tensors:
        collected_tensors[k] = v

  # Handle standalone tensor
  if not collected_tensors and isinstance(res, torch.Tensor):
    collected_tensors["raw"] = res

  if collected_tensors:
    return ModelBenchmarkOutput(ModelBenchmarkOutputType.ALL, collected_tensors)

  # Hard fallback if nothing could be discovered
  return ModelBenchmarkOutput(ModelBenchmarkOutputType.RAW, res)


def _get_device_timings(enable_xprof, session_id) -> tuple[float, float]:
  """Returns the total and average device timings for a given session id."""
  xprof_client = None
  if enable_xprof:
    xprof_client = pt_performance_utils.get_xprof_client()

  total_device_time = -1.0
  avg_device_time = -1.0
  if enable_xprof and xprof_client:
    total_device_time = device_utils.get_max_total_device_time(
        session_id, xprof_client
    )
    if total_device_time != -1.0:
      avg_device_time = (
          total_device_time / pt_benchmark_utils.POST_WARMUP_STEPS.value
      )

  return total_device_time, avg_device_time


def _run_torchax_forward_pass(
    model_jittable: torchax.interop.JittableModule,
    inputs: Any,
    run_mode: pt_common.RunMode,
    enable_xprof: bool,
) -> metrics_lib.PerformanceMetrics:
  """Runs the forward pass benchmark for a TorchAx model."""

  weights = {
      k: v.data if isinstance(v, torch.nn.Parameter) else v
      for k, v in model_jittable.params.items()
  }
  buffers = model_jittable.buffers

  def model_fn(params, buffers, inputs):
    return _call_functional_model(model_jittable, params, buffers, inputs)

  if pt_common.is_torch_compile(run_mode):
    runnable_model = torchax.interop.jax_jit(model_fn)
  else:
    runnable_model = model_fn

  e2e_start = time.perf_counter()

  # Warmup
  warmup_timings = np.zeros(
      pt_benchmark_utils.MIN_WARMUP_STEPS.value, dtype=np.float64
  )
  with pt_benchmark_utils.XprofContext(
      "warmup_run", enable_xprof
  ) as warmup_run_context:
    for i in range(pt_benchmark_utils.MIN_WARMUP_STEPS.value):
      with traceme.TraceMe("Warmup", step_num=i):
        step_start = time.perf_counter()
        out = runnable_model(weights, buffers, inputs)
        _sync_jax_device(out.data)

        warmup_timings[i] = time.perf_counter() - step_start

  first_step_time = warmup_timings[0] if len(warmup_timings) > 0 else 0.0
  warmup_session_xprof_url = None
  if enable_xprof:
    warmup_session_xprof_url = (
        f"http://xprof/?session_id={warmup_run_context.session_id}"
    )

  # Eval
  eval_timings = np.zeros(
      pt_benchmark_utils.POST_WARMUP_STEPS.value, dtype=np.float64
  )
  with pt_benchmark_utils.XprofContext(
      "post_warmup_run", enable_xprof
  ) as post_warmup_run_context:
    for i in range(pt_benchmark_utils.POST_WARMUP_STEPS.value):
      with traceme.TraceMe("Eval", step_num=i):
        step_start = time.perf_counter()
        out = runnable_model(weights, buffers, inputs)
        _sync_jax_device(out.data)
        eval_timings[i] = time.perf_counter() - step_start
  post_warmup_run_session_xprof_url = None
  if enable_xprof:
    post_warmup_run_session_xprof_url = (
        f"http://xprof/?session_id={post_warmup_run_context.session_id}"
    )

  _, avg_device_time = _get_device_timings(
      enable_xprof, post_warmup_run_context.session_id
  )
  peak_device_memory_mb = -1.0
  if enable_xprof:
    xprof_client = pt_performance_utils.get_xprof_client()
    peak_device_memory_mb = device_utils.get_peak_memory_hbm(
        "jax", post_warmup_run_context.session_id, xprof_client
    )

  eval_time = np.mean(eval_timings) if len(eval_timings) > 0 else 0.0

  warmup_overhead = np.sum(warmup_timings) - (eval_time * len(warmup_timings))

  return metrics_lib.PerformanceMetrics(
      num_warmup_steps=len(warmup_timings),
      first_step_time_seconds=first_step_time,
      warmup_overhead_seconds=max(0.0, warmup_overhead),
      post_warmup_step_time_seconds=eval_time,  # pyrefly: ignore[bad-argument-type]
      e2e_wall_time_seconds=time.perf_counter() - e2e_start,
      warmup_session_xprof_url=warmup_session_xprof_url,
      post_warmup_run_session_xprof_url=post_warmup_run_session_xprof_url,
      average_post_warmup_device_time_seconds=avg_device_time,
      peak_device_memory_mb=peak_device_memory_mb,
  )


def _run_torchax_backward_pass(
    model_jittable: torchax.interop.JittableModule,
    inputs: Any,
    run_mode: pt_common.RunMode,
    enable_xprof: bool,
) -> metrics_lib.PerformanceMetrics:
  """Runs the backward pass benchmark for a TorchAx model."""

  weights = {
      k: v.data if isinstance(v, torch.nn.Parameter) else v
      for k, v in model_jittable.params.items()
  }
  buffers = model_jittable.buffers

  # Generate dummy labels
  with torch.no_grad():
    out = _call_functional_model(model_jittable, weights, buffers, inputs)
  output_data = out.data[0] if isinstance(out.data, tuple) else out.data
  if isinstance(output_data, dict):
    labels = {
        k: torch.rand_like(v, device="jax") for k, v in output_data.items()
    }
  else:
    labels = torch.rand_like(output_data, device="jax")

  def loss_fn(outputs, labels):
    if isinstance(outputs, ModelBenchmarkOutput):
      if outputs.return_type == ModelBenchmarkOutputType.LOSS:
        return outputs.data
      outputs_data = (
          outputs.data[0] if isinstance(outputs.data, tuple) else outputs.data
      )
      if isinstance(outputs_data, dict):
        # Accumulate MSE for all discovered tensor components.
        total_loss = 0.0
        for k, v in outputs_data.items():
          if k in labels:
            total_loss = total_loss + torch.mean((v - labels[k]) ** 2)
        return total_loss
      return torch.mean((outputs_data - labels) ** 2)
    outputs_data = outputs[0] if isinstance(outputs, tuple) else outputs
    return torch.mean((outputs_data - labels) ** 2)

  optimizer = optax.adam(0.1)
  opt_state = torchax.interop.call_jax(optimizer.init, weights)

  def model_fn(params, buffers, inputs):
    return _call_functional_model(model_jittable, params, buffers, inputs)

  train_step = torchax.train.make_train_step(model_fn, loss_fn, optimizer)

  if pt_common.is_torch_compile(run_mode):
    runnable_step = torchax.interop.jax_jit(
        train_step, kwargs_for_jax_jit={"donate_argnums": (0, 2)}
    )
  else:
    runnable_step = train_step

  e2e_start = time.perf_counter()

  # Warmup
  warmup_timings = np.zeros(
      pt_benchmark_utils.MIN_WARMUP_STEPS.value, dtype=np.float64
  )
  with pt_benchmark_utils.XprofContext(
      "warmup_run", enable_xprof
  ) as warmup_run_context:
    for i in range(pt_benchmark_utils.MIN_WARMUP_STEPS.value):
      with traceme.TraceMe("Warmup", step_num=i):
        step_start = time.perf_counter()
        loss, weights, opt_state = runnable_step(
            weights, buffers, opt_state, inputs, labels
        )
        _sync_jax_device(loss)
        _sync_jax_device(weights)
        warmup_timings[i] = time.perf_counter() - step_start

  first_step_time = warmup_timings[0] if len(warmup_timings) > 0 else 0.0

  warmup_session_xprof_url = None
  if enable_xprof:
    warmup_session_xprof_url = (
        f"http://xprof/?session_id={warmup_run_context.session_id}"
    )
  # Train
  eval_timings = np.zeros(
      pt_benchmark_utils.POST_WARMUP_STEPS.value, dtype=np.float64
  )
  with pt_benchmark_utils.XprofContext(
      "post_warmup_run", enable_xprof
  ) as post_warmup_run_context:
    for i in range(pt_benchmark_utils.POST_WARMUP_STEPS.value):
      with traceme.TraceMe("Train", step_num=i):
        step_start = time.perf_counter()
        loss, weights, opt_state = runnable_step(
            weights, buffers, opt_state, inputs, labels
        )
        _sync_jax_device(loss)
        _sync_jax_device(weights)
        eval_timings[i] = time.perf_counter() - step_start

  post_warmup_run_session_xprof_url = None
  if enable_xprof:
    post_warmup_run_session_xprof_url = (
        f"http://xprof/?session_id={post_warmup_run_context.session_id}"
    )

  _, avg_device_time = _get_device_timings(
      enable_xprof, post_warmup_run_context.session_id
  )
  peak_device_memory_mb = -1.0
  if enable_xprof:
    xprof_client = pt_performance_utils.get_xprof_client()
    peak_device_memory_mb = device_utils.get_peak_memory_hbm(
        "jax", post_warmup_run_context.session_id, xprof_client
    )

  eval_time = np.mean(eval_timings) if len(eval_timings) > 0 else 0.0

  warmup_overhead = np.sum(warmup_timings) - (eval_time * len(warmup_timings))

  return metrics_lib.PerformanceMetrics(
      num_warmup_steps=len(warmup_timings),
      first_step_time_seconds=first_step_time,
      warmup_overhead_seconds=max(0.0, warmup_overhead),
      post_warmup_step_time_seconds=eval_time,  # pyrefly: ignore[bad-argument-type]
      e2e_wall_time_seconds=time.perf_counter() - e2e_start,
      warmup_session_xprof_url=warmup_session_xprof_url,
      post_warmup_run_session_xprof_url=post_warmup_run_session_xprof_url,
      average_post_warmup_device_time_seconds=avg_device_time,
      peak_device_memory_mb=peak_device_memory_mb,
  )


def prepare_for_torchax(model: torch.nn.Module, inputs):
  """Prepares the model and inputs for TorchAx execution by moving them to JAX.

  This recursively moves parameters/buffers in-place (bypassing the PyTorch C++
  Parameter dispatcher bypass bug and preserving weight tying) and moves
  inputs (out-of-place) returning the JAXified inputs.
  """

  default_jax_device = jax.devices()[0]

  memo = {}
  # Disable DLPack for data conversion to force fallback to numpy,
  # which correctly respects jax.default_device context!
  torchax.default_env().config.use_dlpack_for_data_conversion = False

  with jax.default_device(default_jax_device):

    def _move(module):
      for name, param in list(module.named_parameters(recurse=False)):
        param_id = id(param)
        if param_id in memo:
          module.register_parameter(name, memo[param_id])
        else:
          param_jax_data = param.data.to("jax")
          new_param = torch.nn.Parameter(param_jax_data, param.requires_grad)
          module.register_parameter(name, new_param)
          memo[param_id] = new_param

      for k in dir(module):
        try:
          v = getattr(module, k)
        except:
          continue
        if isinstance(v, torch.Tensor) and not isinstance(
            v, torch.nn.Parameter
        ):
          setattr(module, k, v.to("jax"))

      for child in module.children():
        _move(child)

    _move(model)

    def _move_tensor(x):
      if isinstance(x, torch.Tensor):
        res = x.to("jax")
        assert isinstance(res, torch.Tensor) and isinstance(
            res, torchax.tensor.Tensor
        )
        return res
      return x

    inputs_jax = jax.tree_util.tree_map(_move_tensor, inputs)

  return inputs_jax


def get_model_and_input(
    config: pt_performance_utils.PerformanceBenchmarkConfig,
    cpu_device: torch.device,
    weights_dtype: torch.dtype,
):
  """Gets model and input, handling TorchAx setup."""
  model_and_input = config.model_and_input_factory(
      model_and_input_args=config.model_and_input_args,
      device=cpu_device,
      weights_dtype=weights_dtype,
      is_training=config.is_training,
  )

  model = model_and_input.model
  inputs = model_and_input.example_inputs

  inputs = prepare_for_torchax(model, inputs)

  model_jittable = torchax.interop.JittableModule(model).to("jax")

  return model_jittable, inputs


def run_benchmark(
    config: pt_performance_utils.PerformanceBenchmarkConfig,
    test_method_name: str,
    benchmark_name: str,
    microbenchmark_name: str | None = None,
) -> None:
  """Runs the performance benchmark for a TorchAx model."""

  # TorchAx execution is managed by JAX, which will use the default JAX device
  # (e.g. TPU). We load the model on CPU first to avoid initializing it via
  # the standard PyTorch/XLA backend on TPU.
  cpu_device = torch.device("cpu")
  weights_dtype = pt_performance_utils.get_torch_dtype(
      pt_performance_utils.WEIGHTS_DTYPE.value
  )

  rank = int(os.environ.get("RANK", "0"))
  enable_xprof = pt_performance_utils.ENABLE_XPROF.value and rank == 0

  # Load model and input, handling TorchAx setup
  model_jittable, inputs = get_model_and_input(
      config, cpu_device, weights_dtype
  )

  succeeded = False
  result = None
  exception = None

  try:
    if config.is_training:
      result = _run_torchax_backward_pass(
          model_jittable, inputs, config.run_mode, enable_xprof
      )
    else:
      result = _run_torchax_forward_pass(
          model_jittable, inputs, config.run_mode, enable_xprof
      )
    succeeded = True
  except Exception as e:  # pylint: disable=broad-except
    logging.exception(
        "Performance benchmark failed for %s", test_method_name, e
    )
    exception = e

  if succeeded and result is not None:
    logging.info(
        "Performance Benchmark Results:\n"
        "  Test: %s\n"
        "  benchmark: %s\n"
        "  microbenchmark: %s\n"
        "  run_mode: %s\n"
        "  is_training: %s\n"
        "  warmup_overhead (seconds): %s\n"
        "  average_step_time (seconds): %s\n"
        "  average_post_warmup_device_time (seconds): %s\n"
        "  peak_device_memory (MB): %s\n"
        "  first_step_time (seconds): %s\n"
        "  e2e_wall_time (seconds): %s\n"
        "  warmup_session_xprof_url: %s\n"
        "  post_warmup_run_session_xprof_url: %s",
        test_method_name,
        benchmark_name,
        microbenchmark_name,
        config.run_mode.value,
        config.is_training,
        result.warmup_overhead_seconds,
        result.post_warmup_step_time_seconds,
        result.average_post_warmup_device_time_seconds,
        result.peak_device_memory_mb,
        result.first_step_time_seconds,
        result.e2e_wall_time_seconds,
        result.warmup_session_xprof_url,
        result.post_warmup_run_session_xprof_url,
    )

  if pt_benchmark_utils.MLCOMPASS_TRACKING_ID.value:
    mlcompass_utils.export_to_mlcompass(
        pt_common.PLATFORM.value,
        result,
        pt_benchmark_utils.BASE_CL.value,
        pt_benchmark_utils.MLCOMPASS_TRACKING_ID.value,
        pt_benchmark_utils.MLCOMPASS_EXECUTION_MODE.value,
        test_method_name=test_method_name,
        benchmark_name=benchmark_name,
        microbenchmark_name=microbenchmark_name,
        succeeded=succeeded,
        pending_cl=pt_benchmark_utils.PENDING_CL.value,
        benchmark_group=pt_benchmark_utils.BENCHMARK_GROUP.value,
    )

  if exception:
    raise exception
