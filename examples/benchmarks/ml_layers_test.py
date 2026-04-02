# Copyright 2025 Google LLC
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

test_var123 = 123

"""Micro benchmarks for common ML layers."""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
import json
import os
import pprint
import sys
import time
from typing import Any, TypeAlias

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import flax
from flax import nnx
import jax
import jax.numpy as jnp
import torch
import torch._inductor.config as inductor_config
from torch.nn import attention
from torch.utils import tensorboard
from torch_tpu import api
from torch_tpu._internal import compile as torch_tpu_compile
from torch_tpu._internal import execution_mode
from torch_tpu._internal import sync
from torch_tpu._internal.utils import benchmarking
from tests import ops_test_data

generate_configs_for_parameterized = (
    ops_test_data.generate_configs_for_parameterized
)

from torch_tpu._internal.shims.xprof import traceme
from torch_tpu._internal.shims.xprof import xprof_session

Metric: TypeAlias = dict[str, str | float]
Metrics: TypeAlias = Sequence[Metric]

_NUM_WARMUP_STEPS = flags.DEFINE_integer(
    "num_warmup_steps", 5, "Numer of warm-up steps."
)

_NUM_STEPS = flags.DEFINE_integer(
    "num_steps", 100, "Numer of post-warm up steps."
)

_DEVICE = flags.DEFINE_enum(
    "device",
    None,
    ["tpu", "cuda", "xla_cuda"],
    required=True,
    help="Accelerator to test.",
)

_COMPILE = flags.DEFINE_bool(
    "compile",
    False,
    "Whether to use compile the model.",
)

_EAGER_MODE = flags.DEFINE_enum(
    "eager_mode",
    "DEFAULT",
    ["DEFAULT", "OPTIMIZED", "DEFER_NEVER", "DEFER_NEVER_AND_LAUNCH_BLOCKING"],
    "Eager mode for the model.",
)

_FRAMEWORK = flags.DEFINE_enum(
    "framework",
    None,
    ["pytorch", "jax"],
    required=True,
    help="Framework to test.",
)

_OUTPUT_FORMAT = flags.DEFINE_enum(
    "output_format",
    "text",
    ["text", "json"],
    help="Output format.",
)

_CAPTURE_DELIM = flags.DEFINE_string(
    "capture_delim",
    "@",
    "Surrounds structured output so that it may be captured by build"
    " orchestration tools listening on this tests console output. Has no effect"
    " in text output mode.",
)


def get_eager_mode() -> execution_mode.EagerMode:
  if _EAGER_MODE.value == "DEFAULT":
    return execution_mode.EagerMode.DEFAULT
  elif _EAGER_MODE.value == "OPTIMIZED":
    return execution_mode.EagerMode.OPTIMIZED
  elif _EAGER_MODE.value == "DEFER_NEVER":
    return execution_mode.EagerMode.DEFER_NEVER
  elif _EAGER_MODE.value == "DEFER_NEVER_AND_LAUNCH_BLOCKING":
    return execution_mode.EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING
  else:
    raise ValueError(f"Unsupported defer mode: {_EAGER_MODE.value}")


def get_mode() -> str:
  if _COMPILE.value:
    return "compiled"
  else:
    return f"{execution_mode.get_eager_mode()}"


def get_torch_device() -> torch.device:
  if _DEVICE.value == "tpu":
    return api.tpu_device()
  elif _DEVICE.value == "cuda":
    return torch.device("cuda")
  elif _DEVICE.value == "xla_cuda":
    return api._xla_cuda_device()
  else:
    raise ValueError(f"Unsupported device: {_DEVICE.value}")


def get_jax_device() -> jax.Device:
  return jax.devices()[0]


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
    torch.cuda.synchronize()


def clear_compilation_cache() -> None:
  if _DEVICE.value in ("tpu", "xla_cuda"):
    getattr(torch, _DEVICE.value)._clear_cache()


def torch_compile_model(model):
  if _DEVICE.value == "cuda":
    model = torch.compile(model)
  elif _DEVICE.value in ("tpu", "xla_cuda"):
    model = torch.compile(
        model, dynamic=False, backend=torch_tpu_compile.TpuBackend()
    )
  return model


def pt2jax_dtype(dtype: torch.dtype) -> jnp.dtype:
  if dtype == torch.bfloat16:
    return jnp.bfloat16
  elif dtype == torch.float32:
    return jnp.float32
  else:
    raise ValueError(f"Unsupported dtype: {dtype}")


def is_pytorch_framework() -> bool:
  return _FRAMEWORK.value == "pytorch"


def is_jax_framework() -> bool:
  return _FRAMEWORK.value == "jax"


def reset_torch_cuda_peak_memory():
  if _DEVICE.value == "cuda":
    torch.cuda.memory.reset_peak_memory_stats()


def get_torch_cuda_peak_memory() -> dict[str, Any]:
  if _DEVICE.value == "cuda":
    peak_memory_usage_mb = torch.cuda.memory.max_memory_allocated() / 1_048_576
    print(
        f"ACTUAL CUDA MEMORY USAGE: {peak_memory_usage_mb:.2f} MB",
        flush=True,
    )
    return {"peak_memory_usage_mb": peak_memory_usage_mb}
  return {}


class MlLayersTest(parameterized.TestCase):
  all_metrics = []

  @classmethod
  def tearDownClass(cls):
    mode = get_mode()

    results = {}
    for m in cls.all_metrics:
      # Sort config for deterministic key
      c_str = str(dict(sorted(m["config"].items(), key=lambda x: x[0])))
      key = (m["model_name"], c_str)
      if key not in results:
        results[key] = {}
      results[key][m["mode"]] = m

    if _OUTPUT_FORMAT.value == "json":
      json_results = []
      for (model_name, _), modes in sorted(results.items()):
        fwd_metrics = modes.get("eval", {})
        bwd_metrics = modes.get("train", {})

        # Use config from whatever mode is available
        config = fwd_metrics.get("config") or bwd_metrics.get("config") or {}

        # Helper to safely convert ms times
        def get_time_ms(t):
          return t * 1e3 if t is not None else None

        # Build performance object
        performance = {
            "forward": {
                "warmup_time_ms": get_time_ms(fwd_metrics.get("warmup_time")),
                "evaluation_time_ms": get_time_ms(
                    fwd_metrics.get("evaluation_time")
                ),
            },
            "backward": {
                "warmup_time_ms": get_time_ms(bwd_metrics.get("warmup_time")),
                "evaluation_time_ms": get_time_ms(
                    bwd_metrics.get("evaluation_time")
                ),
            },
        }

        xprof_url = (
            fwd_metrics.get("xprof_url")
            if fwd_metrics
            else (bwd_metrics.get("xprof_url") if bwd_metrics else None)
        )

        json_results.append({
            "model": model_name,
            "configuration": config,
            "performance": performance,
            "xprof_url": xprof_url,
        })

      output_data = {
          "metadata": {
              "framework": _FRAMEWORK.value,
              "device": _DEVICE.value,
              "mode": mode,
          },
          "results": json_results,
      }

      def json_default(obj):
        if isinstance(obj, (torch.dtype, jnp.dtype)):
          return str(obj)
        # Handle other potential non-serializable types if needed
        return str(obj)

      if _CAPTURE_DELIM.value:
        json_out = f"{_CAPTURE_DELIM.value}%s{_CAPTURE_DELIM.value}"
      else:
        json_out = "%s"
      print(json_out % json.dumps(output_data, default=json_default, indent=2))

    else:
      # Text output (original behavior)
      print(f"All Metrics for:")
      print(f"\tframework: {_FRAMEWORK.value}")
      print(f"\tdevice: {_DEVICE.value}")
      print(f"\tmode: {mode}")

      # Define column widths
      w_model = 20
      w_config = 80
      if results:
        max_config_len = max(len(c) for _, c in results.keys())
        w_config = max(w_config, max_config_len)
      w_num = 12
      w_xprof = 50

      model_column = f"{'Model':<{w_model}}"
      cfg_column = f"{'Configuration':<{w_config}}"
      fwd_w_column = f"{'Fwd Warmup':<{w_num}}"
      fwd_e_column = f"{'Fwd Eval':<{w_num}} "
      bwd_w_column = f"{'Bwd Warmup':<{w_num}}"
      bwd_e_column = f"{'Bwd Eval':<{w_num}}"
      xprof_column = f"{'XProf':<{w_xprof}}"
      columns = [
          model_column,
          cfg_column,
          fwd_w_column,
          fwd_e_column,
          bwd_w_column,
          bwd_e_column,
          xprof_column,
      ]

      splits = []
      for c in columns:
        splits.append("-" * len(c))

      print("")
      print(" | ".join(columns))
      print("-|-".join(splits))

      for (model, config), modes in sorted(results.items()):
        fwd_metrics = modes.get("eval", {})
        bwd_metrics = modes.get("train", {})

        def fmt_time(t):
          return f"{t*1e3:.2f} ms" if t is not None else "N/A"

        fw_w = fmt_time(fwd_metrics.get("warmup_time"))
        fw_e = fmt_time(fwd_metrics.get("evaluation_time"))
        bw_w = fmt_time(bwd_metrics.get("warmup_time"))
        bw_e = fmt_time(bwd_metrics.get("evaluation_time"))

        # Since xprof_url is the same for both eval and train in a _run_model_tests call,
        # picking it from either is fine. Prefer 'eval' as it's always present.
        xprof = (
            fwd_metrics.get("xprof_url", "N/A")
            if fwd_metrics
            else (bwd_metrics.get("xprof_url", "N/A") if bwd_metrics else "N/A")
        )

        print(
            f"{model:<{w_model}} | {config:<{w_config}} | {fw_w:<{w_num}} |"
            f" {fw_e:<{w_num}} | {bw_w:<{w_num}} | {bw_e:<{w_num}} |"
            f" {xprof:<{w_xprof}}"
        )
      print("")

    super().tearDownClass()

  def setUp(self):
    super().setUp()

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

    execution_mode.set_eager_mode(get_eager_mode())

  def tearDown(self):
    super().tearDown()

  def _run_torch_forward_pass(
      self,
      model: torch.nn.Module,
      inputs: torch.Tensor,
      device: torch.device,
  ) -> Metric:
    reset_torch_cuda_peak_memory()
    clear_compilation_cache()

    if _COMPILE.value:
      with traceme.TraceMe("Compile"):
        model = torch_compile_model(model)

    runnable_model = model.to(device)
    runnable_model.eval()

    # Warm up the model.
    warmup_start_time = time.time()
    for i in range(_NUM_WARMUP_STEPS.value):
      with traceme.TraceMe("Warmup", step_num=i):
        output = runnable_model(inputs)
        sync_device(output)
    warmup_end_time = time.time()
    warmup_time = (
        warmup_end_time - warmup_start_time
    ) / _NUM_WARMUP_STEPS.value
    logging.info("Warmup average step time: %fms", 1e3 * warmup_time)

    # Run the inference steps.
    eval_start_time = time.time()
    for i in range(_NUM_STEPS.value):
      with traceme.TraceMe("Eval", step_num=i):
        output = runnable_model(inputs)
        sync_device(output)
    eval_end_time = time.time()
    evaluation_time = (eval_end_time - eval_start_time) / _NUM_STEPS.value
    logging.info("Eval average step time: %fms", 1e3 * evaluation_time)

    return {
        "warmup_time": warmup_time,
        "evaluation_time": evaluation_time,
        "mode": "eval",
        **get_torch_cuda_peak_memory(),
    }

  def _run_jax_forward_pass(
      self,
      model: flax.nnx.Module,
      inputs: jax.Array,
      device: jax.Device,
  ) -> Metric:
    model.eval()
    if _COMPILE.value:
      with traceme.TraceMe("Compile"):
        runnable_model = flax.nnx.jit(model)
    else:
      runnable_model = model
    inputs_on_device = inputs.to_device(device)

    def sync_jax_device(x):
      jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), x)

    # Warm up the model.
    warmup_start_time = time.time()
    for i in range(_NUM_WARMUP_STEPS.value):
      with traceme.TraceMe("Warmup", step_num=i):
        output = runnable_model(inputs_on_device)
        sync_jax_device(output)
    warmup_end_time = time.time()
    warmup_time = (
        warmup_end_time - warmup_start_time
    ) / _NUM_WARMUP_STEPS.value
    logging.info("Warmup average step time: %fms", 1e3 * warmup_time)

    # Run the training steps.
    eval_start_time = time.time()
    for i in range(_NUM_STEPS.value):
      with traceme.TraceMe("Eval", step_num=i):
        output = runnable_model(inputs_on_device)
        sync_jax_device(output)
    eval_end_time = time.time()
    evaluation_time = (eval_end_time - eval_start_time) / _NUM_STEPS.value
    logging.info("Eval average step time: %fms", 1e3 * evaluation_time)

    return {
        "warmup_time": warmup_time,
        "evaluation_time": evaluation_time,
        "mode": "eval",
    }

  def _run_torch_backward_pass(
      self,
      model: torch.nn.Module,
      inputs: torch.Tensor,
      device: torch.device,
  ) -> Metric:
    clear_compilation_cache()
    reset_torch_cuda_peak_memory()

    model_on_device = model.to(device)
    model_on_device.train()

    def train_step(x):
      y_pred = model_on_device(x)
      loss = torch.mean(y_pred)
      loss.backward()
      # We call detach() on the loss tensor to allow Dynamo to trace through the
      # backward pass and generate a single graph via torch.compile.
      return loss.detach()

    if _COMPILE.value:
      with traceme.TraceMe("Compile"):
        runnable_model = torch_compile_model(train_step)
    else:
      runnable_model = train_step
    inputs_on_device = inputs.to(device)

    def sync_loss_and_grads(loss_val, model):
      sync_device(loss_val)
      for _, p in model.named_parameters():
        if p.grad is not None:
          sync_device(p.grad)

    # Warm up the model.
    warmup_start_time = time.time()
    for i in range(_NUM_WARMUP_STEPS.value):
      with traceme.TraceMe("Warmup", step_num=i):
        model_on_device.zero_grad()
        loss_val = runnable_model(inputs_on_device)
        sync_loss_and_grads(loss_val, model_on_device)
    warmup_end_time = time.time()
    warmup_time = (
        warmup_end_time - warmup_start_time
    ) / _NUM_WARMUP_STEPS.value
    logging.info("Warmup average step time: %fms", 1e3 * warmup_time)

    # Run training steps.
    eval_start_time = time.time()
    for i in range(_NUM_STEPS.value):
      with traceme.TraceMe("Train", step_num=i):
        model_on_device.zero_grad()
        loss_val = runnable_model(inputs_on_device)
        sync_loss_and_grads(loss_val, model_on_device)
    eval_end_time = time.time()
    evaluation_time = (eval_end_time - eval_start_time) / _NUM_STEPS.value
    logging.info("Eval average step time: %fms", 1e3 * evaluation_time)

    return {
        "warmup_time": warmup_time,
        "evaluation_time": evaluation_time,
        "mode": "train",
        **get_torch_cuda_peak_memory(),
    }

  def _run_jax_backward_pass(
      self,
      model: flax.nnx.Module,
      inputs: jax.Array,
      device: jax.Device,
  ) -> Metric:
    model.train()

    def grad_step(model, x):
      def loss_fn(model, x):
        y_pred = model(x)
        return jnp.mean(y_pred)

      loss_val, grads = flax.nnx.value_and_grad(loss_fn)(model, x)
      return loss_val, grads

    if _COMPILE.value:
      with traceme.TraceMe("Compile"):
        grad_step = flax.nnx.jit(grad_step)
    runnable_model = lambda x: grad_step(model, x)
    inputs_on_device = inputs.to_device(device)

    # Warm up the model.
    warmup_start_time = time.time()
    for i in range(_NUM_WARMUP_STEPS.value):
      with traceme.TraceMe("Warmup", step_num=i):
        loss_val, grads = runnable_model(inputs_on_device)
        loss_val.block_until_ready()
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), grads)
    warmup_end_time = time.time()
    warmup_time = (
        warmup_end_time - warmup_start_time
    ) / _NUM_WARMUP_STEPS.value
    logging.info("Warmup average step time: %fms", 1e3 * warmup_time)

    # Run training steps.
    eval_start_time = time.time()
    for i in range(_NUM_STEPS.value):
      with traceme.TraceMe("Train", step_num=i):
        loss_val, grads = runnable_model(inputs_on_device)
        loss_val.block_until_ready()
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), grads)
    eval_end_time = time.time()
    evaluation_time = (eval_end_time - eval_start_time) / _NUM_STEPS.value
    logging.info("Eval average step time: %fms", 1e3 * evaluation_time)

    return {
        "warmup_time": warmup_time,
        "evaluation_time": evaluation_time,
        "mode": "train",
    }

  def _run_jax_passes(
      self,
      test_name: str,
      config: Any,
      model_builder: Callable[[Any], torch.nn.Module | flax.nnx.Module],
      inputs_builder: Callable[..., torch.Tensor | jax.Array],
      skip_bw_pass: bool = False,
  ) -> Metrics:
    # Reset the JAX caches so that it won't reuse previous compilation from previously run test
    # cases.
    jax.clear_caches()

    if _DEVICE.value != "tpu":
      # JAX is only tested on TPU in this benchmark.
      raise ValueError(f"JAX tests not supported for: {_DEVICE.value}")

    device = get_jax_device()

    pt_seed = torch.initial_seed()
    jax_key = jax.random.PRNGKey(pt_seed)
    model_fwd = model_builder(config)

    logging.info(
        "======== Running FW pass for %s with config %s ========",
        test_name,
        pprint.pformat(asdict(config)),
    )
    jax_key, fwd_subkey = jax.random.split(jax_key)
    fwd_inputs = inputs_builder(config, fwd_subkey)
    fwd_metric = self._run_jax_forward_pass(model_fwd, fwd_inputs, device)

    if skip_bw_pass:
      return [fwd_metric]

    model_bwd = model_builder(config)

    logging.info(
        "======== Running FW+BW pass for %s with config %s ========",
        test_name,
        pprint.pformat(asdict(config)),
    )
    _, bw_subkey = jax.random.split(jax_key)
    bw_inputs = inputs_builder(config, bw_subkey)
    bw_metric = self._run_jax_backward_pass(model_bwd, bw_inputs, device)

    return [fwd_metric, bw_metric]

  def _run_torch_passes(
      self,
      test_name: str,
      config: Any,
      model_builder: Callable[[Any], torch.nn.Module | flax.nnx.Module],
      inputs_builder: Callable[..., torch.Tensor | jax.Array],
      skip_bw_pass: bool = False,
  ) -> Metrics:
    # Reset the Torch Dynamo cache so that it won't reuse previous compilation from previously run
    # test cases.
    torch.compiler.reset()

    device = get_torch_device()

    model_fwd = model_builder(config)
    logging.info(
        "======== Running FW pass for %s with config %s ========",
        test_name,
        pprint.pformat(asdict(config)),
    )
    with torch.inference_mode():
      fwd_inputs = inputs_builder(config)
      fwd_metric = self._run_torch_forward_pass(model_fwd, fwd_inputs, device)

    if skip_bw_pass:
      return [fwd_metric]

    model_bwd = model_builder(config)
    logging.info(
        "======== Running FW+BW pass for %s with config %s ========",
        test_name,
        pprint.pformat(asdict(config)),
    )
    bw_inputs = inputs_builder(config)
    bw_metric = self._run_torch_backward_pass(model_bwd, bw_inputs, device)

    return [fwd_metric, bw_metric]

  def _run_model_tests(
      self,
      test_name: str,
      config: Any,
      model_builder: Callable[[Any], torch.nn.Module | flax.nnx.Module],
      inputs_builder: Callable[..., torch.Tensor | jax.Array],
      *,
      skip_bw_pass=False,
      is_jax=False,
  ) -> Metrics:
    metrics = []
    session = xprof_session.XprofSession()
    session.start_session(host_trace_level=3, enable_python_tracer=True)
    try:
      if is_jax:
        metrics.extend(
            self._run_jax_passes(
                test_name,
                config,
                model_builder,
                inputs_builder,
                skip_bw_pass,
            )
        )
      else:
        metrics.extend(
            self._run_torch_passes(
                test_name,
                config,
                model_builder,
                inputs_builder,
                skip_bw_pass,
            )
        )

    finally:
      xprof_url = session.end_session_and_get_url()
      MlLayersTest.all_metrics.extend(
          m
          | {
              "xprof_url": xprof_url,
              "model_name": test_name,
              "config": asdict(config),
          }
          for m in metrics
      )
    return metrics

  # ############################################################################
  # BatchNorm1d layer tests
  # ############################################################################

  @dataclass
  class _BatchNormConfig:
    batch_size: int
    seq_len: int
    num_features: int
    dtype: torch.dtype

  _batch_norm_configs = (
      # Default config for smoke test.
      _BatchNormConfig(
          batch_size=1,
          seq_len=128,
          num_features=128,
          dtype=torch.bfloat16,
      ),
      # Larger configs.
      _BatchNormConfig(
          batch_size=32,
          seq_len=8192,
          num_features=8192,
          dtype=torch.bfloat16,
      ),
      # High-batch, more TPU friendly shape.
      _BatchNormConfig(
          batch_size=2056,
          seq_len=512,
          num_features=1024,
          dtype=torch.bfloat16,
      ),
  )

  @parameterized.named_parameters(
      generate_configs_for_parameterized(_batch_norm_configs)
  )
  def test_nn_BatchNorm1d(self, config):
    """Benchmark torch.nn.BatchNorm1d with random data."""
    if not is_pytorch_framework():
      self.skipTest("PyTorch not enabled")

    class Model(torch.nn.Module):

      def __init__(self, num_features: int, dtype: torch.dtype):
        super().__init__()
        self.batchnorm = torch.nn.BatchNorm1d(num_features, dtype=dtype)

      def forward(self, x) -> torch.Tensor:
        return self.batchnorm(x)

    self._run_model_tests(
        "nn.BatchNorm1d",
        config,
        lambda c: Model(c.num_features, c.dtype),
        lambda c: torch.randn(
            (c.batch_size, c.num_features, c.seq_len),
            dtype=c.dtype,
            device=get_torch_device(),
        ),
    )

  @parameterized.named_parameters(
      generate_configs_for_parameterized(_batch_norm_configs)
  )
  def test_nnx_BatchNorm1d(self, config):
    """Benchmark flax.nnx.BatchNorm with random data."""
    if not is_jax_framework():
      self.skipTest("JAX not enabled")

    class Model(flax.nnx.Module):

      def __init__(self, num_features: int, dtype: jnp.dtype, rngs):
        super().__init__()
        self.batchnorm = flax.nnx.BatchNorm(
            num_features=num_features,
            use_running_average=False,
            dtype=dtype,
            rngs=rngs,
        )

      def __call__(self, x) -> jax.Array:
        return self.batchnorm(x)

    self._run_model_tests(
        "nnx.BatchNorm1d",
        config,
        lambda c: Model(
            c.num_features, pt2jax_dtype(c.dtype), flax.nnx.Rngs(0)
        ),
        lambda c, key: jax.random.normal(
            key,
            (c.batch_size, c.seq_len, c.num_features),
            dtype=pt2jax_dtype(c.dtype),
        ),
        is_jax=True,
    )

  # ############################################################################
  # LayerNorm layer tests
  # ############################################################################

  @dataclass
  class _LayerNormConfig:
    _shape: tuple[int, ...]
    dtype: torch.dtype
    num_normalized_dims: int = 1

    @property
    def shape(self):
      return self._shape

    @property
    def normalized_shape(self):
      return self.shape[-self.num_normalized_dims :]

    @property
    def reduction_axes(self):
      return range(len(self.shape) - self.num_normalized_dims, len(self.shape))

    @property
    def num_features(self):
      return self.shape[-1]

  _layer_norm_configs = (
      # Default config for smoke test.
      _LayerNormConfig(
          _shape=(1, 128, 128),
          dtype=torch.bfloat16,
      ),
      # Larger configs.
      _LayerNormConfig(
          _shape=(32, 8192, 8192),
          dtype=torch.bfloat16,
      ),
      _LayerNormConfig(
          _shape=(32, 8192, 8192),
          dtype=torch.bfloat16,
          num_normalized_dims=2,
      ),
  )

  @parameterized.named_parameters(
      generate_configs_for_parameterized(_layer_norm_configs)
  )
  def test_nn_LayerNorm(self, config):
    """Benchmark torch.nn.LayerNorm with random data."""
    if not is_pytorch_framework():
      self.skipTest("PyTorch not enabled")

    class Model(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(
            config.normalized_shape, dtype=config.dtype
        )

      def forward(self, x) -> torch.Tensor:
        return self.layernorm(x)

    self._run_model_tests(
        "nn.LayerNorm",
        config,
        lambda c: Model(),
        lambda c: torch.randn(
            c.shape,
            dtype=c.dtype,
            device=get_torch_device(),
        ),
    )

  @parameterized.named_parameters(
      generate_configs_for_parameterized(_layer_norm_configs)
  )
  def test_nnx_LayerNorm(self, config):
    """Benchmark flax.nnx.LayerNorm with random data."""
    if not is_jax_framework():
      self.skipTest("JAX not enabled")

    class Model(flax.nnx.Module):

      def __init__(self):
        super().__init__()
        self.layernorm = flax.nnx.LayerNorm(
            num_features=config.num_features,
            dtype=pt2jax_dtype(config.dtype),
            rngs=flax.nnx.Rngs(0),
            reduction_axes=config.reduction_axes,
        )

      def __call__(self, x) -> jax.Array:
        return self.layernorm(x)

    self._run_model_tests(
        "nnx.LayerNorm",
        config,
        lambda c: Model(),
        lambda c, key: jax.random.normal(
            key,
            c.shape,
            dtype=pt2jax_dtype(c.dtype),
        ),
        is_jax=True,
    )

  # ############################################################################
  # Linear layer tests
  # ############################################################################

  @dataclass
  class _LinearConfig:
    batch_size: int
    seq_len: int
    in_features: int
    out_features: int
    dtype: torch.dtype

  _linear_configs = (
      # Default config for smoke test.
      _LinearConfig(
          batch_size=1,
          seq_len=128,
          in_features=128,
          out_features=128,
          dtype=torch.bfloat16,
      ),
      # Configs for Llama3 70B for MLP layers
      _LinearConfig(
          batch_size=1,
          seq_len=8192,
          in_features=8192,
          out_features=28672,
          dtype=torch.bfloat16,
      ),
      _LinearConfig(
          batch_size=1,
          seq_len=8192,
          in_features=28672,
          out_features=8192,
          dtype=torch.bfloat16,
      ),
      _LinearConfig(
          batch_size=32,
          seq_len=8192,
          in_features=8192,
          out_features=28672,
          dtype=torch.bfloat16,
      ),
      _LinearConfig(
          batch_size=32,
          seq_len=8192,
          in_features=28672,
          out_features=8192,
          dtype=torch.bfloat16,
      ),
      # Configs for Qwen3 480B MLP layers
      _LinearConfig(
          batch_size=1,
          seq_len=8192,
          in_features=6144,
          out_features=2560,
          dtype=torch.bfloat16,
      ),
      _LinearConfig(
          batch_size=1,
          seq_len=8192,
          in_features=2560,
          out_features=6144,
          dtype=torch.bfloat16,
      ),
      _LinearConfig(
          batch_size=32,
          seq_len=8192,
          in_features=6144,
          out_features=2560,
          dtype=torch.bfloat16,
      ),
      _LinearConfig(
          batch_size=32,
          seq_len=8192,
          in_features=2560,
          out_features=6144,
          dtype=torch.bfloat16,
      ),
      # Configs for Gemma3 27B MLP layers
      _LinearConfig(
          batch_size=1,
          seq_len=8192,
          in_features=4608,
          out_features=36864,
          dtype=torch.bfloat16,
      ),
      _LinearConfig(
          batch_size=1,
          seq_len=8192,
          in_features=36864,
          out_features=4608,
          dtype=torch.bfloat16,
      ),
      _LinearConfig(
          batch_size=32,
          seq_len=8192,
          in_features=4608,
          out_features=36864,
          dtype=torch.bfloat16,
      ),
      _LinearConfig(
          batch_size=32,
          seq_len=8192,
          in_features=36864,
          out_features=4608,
          dtype=torch.bfloat16,
      ),
  )

  @parameterized.named_parameters(
      generate_configs_for_parameterized(_linear_configs)
  )
  def test_nn_Linear(self, config):
    """Benchmark torch.nn.Linear with random data."""
    if not is_pytorch_framework():
      self.skipTest("PyTorch not enabled")

    class Model(torch.nn.Module):

      def __init__(
          self, in_features: int, out_features: int, dtype: torch.dtype
      ):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, dtype=dtype)

      def forward(self, x) -> torch.Tensor:
        return self.linear(x)

    self._run_model_tests(
        "nn.Linear",
        config,
        lambda c: Model(c.in_features, c.out_features, c.dtype),
        lambda c: torch.randn(
            (c.batch_size, c.seq_len, c.in_features),
            dtype=c.dtype,
            device=get_torch_device(),
        ),
    )

  @parameterized.named_parameters(
      generate_configs_for_parameterized(_linear_configs)
  )
  def test_nnx_Linear(self, config):
    """Benchmark flax.nnx.Linear with random data."""
    if not is_jax_framework():
      self.skipTest("JAX not enabled")

    class Model(flax.nnx.Module):

      def __init__(
          self, in_features: int, out_features: int, dtype: jnp.dtype, rngs
      ):
        super().__init__()
        # To match torch.nn.Linear bias initialization:
        # bound = 1 / sqrt(in_features)
        # bias ~ U(-bound, bound)
        bound = 1 / in_features**0.5
        self.linear = flax.nnx.Linear(
            in_features,
            out_features,
            dtype=dtype,
            rngs=rngs,
            bias_init=jax.nn.initializers.uniform(scale=bound),
        )

      def __call__(self, x) -> jax.Array:
        return self.linear(x)

    self._run_model_tests(
        "nnx.Linear",
        config,
        lambda c: Model(
            c.in_features,
            c.out_features,
            pt2jax_dtype(c.dtype),
            flax.nnx.Rngs(0),
        ),
        lambda c, key: jax.random.normal(
            key,
            (c.batch_size, c.seq_len, c.in_features),
            pt2jax_dtype(c.dtype),
        ),
        is_jax=True,
    )

  # ############################################################################
  # Conv2d layer tests
  # ############################################################################

  @dataclass
  class _Conv2dConfig:
    batch_size: int
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    height: int
    width: int
    dtype: torch.dtype

  _conv2d_configs = (
      # Default config for smoke test.
      _Conv2dConfig(
          batch_size=1,
          in_channels=2,
          out_channels=4,
          kernel_size=3,
          stride=1,
          padding=1,
          height=128,
          width=128,
          dtype=torch.bfloat16,
      ),
      # Larger configs.
      _Conv2dConfig(
          batch_size=128,
          in_channels=32,
          out_channels=64,
          kernel_size=3,
          stride=1,
          padding=1,
          height=256,
          width=256,
          dtype=torch.bfloat16,
      ),
  )

  @parameterized.named_parameters(
      generate_configs_for_parameterized(_conv2d_configs)
  )
  def test_nn_Conv2d(self, config):
    """Benchmark torch.nn.Conv2d with random data."""
    if not is_pytorch_framework():
      self.skipTest("PyTorch not enabled")

    class Model(torch.nn.Module):

      def __init__(
          self,
          in_channels: int,
          out_channels: int,
          kernel_size: int,
          stride: int,
          padding: int,
          dtype: torch.dtype,
      ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dtype=dtype,
        )

      def forward(self, x) -> torch.Tensor:
        return self.conv(x)

    self._run_model_tests(
        "nn.Conv2d",
        config,
        lambda c: Model(
            c.in_channels,
            c.out_channels,
            c.kernel_size,
            c.stride,
            c.padding,
            c.dtype,
        ),
        lambda c: torch.randn(
            (c.batch_size, c.in_channels, c.height, c.width),
            dtype=c.dtype,
            device=get_torch_device(),
        ),
    )

  @parameterized.named_parameters(
      generate_configs_for_parameterized(_conv2d_configs)
  )
  def test_nnx_Conv2d(self, config):
    """Benchmark flax.nnx.Conv with random data."""
    if not is_jax_framework():
      self.skipTest("JAX not enabled")

    class Model(flax.nnx.Module):

      def __init__(
          self,
          in_channels: int,
          out_channels: int,
          kernel_size: int,
          stride: int,
          padding: int,
          dtype: jnp.dtype,
          rngs: flax.nnx.Rngs,
      ):
        super().__init__()
        # To match torch.nn.Conv2d bias initialization:
        # bound = 1 / sqrt(in_channels * kernel_size * kernel_size)
        # bias ~ U(-bound, bound)
        bound = 1 / (in_channels * kernel_size**2) ** 0.5
        self.conv = flax.nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride),
            padding=padding,
            dtype=dtype,
            rngs=rngs,
            bias_init=jax.nn.initializers.uniform(scale=bound),
        )

      def __call__(self, x) -> jax.Array:
        return self.conv(x)

    self._run_model_tests(
        "nnx.Conv2d",
        config,
        lambda c: Model(
            c.in_channels,
            c.out_channels,
            c.kernel_size,
            c.stride,
            c.padding,
            pt2jax_dtype(c.dtype),
            flax.nnx.Rngs(0),
        ),
        lambda c, key: jax.random.normal(
            key,
            (c.batch_size, c.height, c.width, c.in_channels),
            dtype=pt2jax_dtype(c.dtype),
        ),
        is_jax=True,
    )

  # ############################################################################
  # RMSNorm layer tests
  # ############################################################################

  @dataclass
  class _RmsNormConfig:
    batch_size: int
    seq_len: int
    num_features: int
    dtype: torch.dtype

  _rms_norm_configs = (
      # Default config for smoke test.
      _RmsNormConfig(
          batch_size=1,
          seq_len=128,
          num_features=128,
          dtype=torch.bfloat16,
      ),
      # Configs for Llama3 70B for MLP layers
      _RmsNormConfig(
          batch_size=1,
          seq_len=8192,
          num_features=8192,
          dtype=torch.bfloat16,
      ),
      _RmsNormConfig(
          batch_size=32,
          seq_len=8192,
          num_features=8192,
          dtype=torch.bfloat16,
      ),
      # Configs for Qwen3 480B MLP layers
      _RmsNormConfig(
          batch_size=1,
          seq_len=8192,
          num_features=6144,
          dtype=torch.bfloat16,
      ),
      _RmsNormConfig(
          batch_size=32,
          seq_len=8192,
          num_features=6144,
          dtype=torch.bfloat16,
      ),
      # Configs for Gemma3 27B MLP layers
      _RmsNormConfig(
          batch_size=1,
          seq_len=8192,
          num_features=4608,
          dtype=torch.bfloat16,
      ),
      _RmsNormConfig(
          batch_size=32,
          seq_len=8192,
          num_features=4608,
          dtype=torch.bfloat16,
      ),
  )

  @parameterized.named_parameters(
      generate_configs_for_parameterized(_rms_norm_configs)
  )
  def test_nn_RMSNorm(self, config):
    """Benchmark torch.nn.RMSNorm with random data."""
    if not is_pytorch_framework():
      self.skipTest("PyTorch not enabled")

    class Model(torch.nn.Module):

      def __init__(self, num_features: torch.Size, dtype: torch.dtype):
        super().__init__()
        self.rmsnorm = torch.nn.RMSNorm(num_features, dtype=dtype)

      def forward(self, x) -> torch.Tensor:
        return self.rmsnorm(x)

    self._run_model_tests(
        "nn.RMSNorm",
        config,
        lambda c: Model(c.num_features, c.dtype),
        lambda c: torch.randn(
            (c.batch_size, c.seq_len, c.num_features),
            dtype=c.dtype,
            device=get_torch_device(),
        ),
    )

  @parameterized.named_parameters(
      generate_configs_for_parameterized(_rms_norm_configs)
  )
  def test_nnx_RMSNorm(self, config):
    """Benchmark flax.nnx.RMSNorm with random data."""
    if not is_jax_framework():
      self.skipTest("JAX not enabled")

    class Model(flax.nnx.Module):

      def __init__(self, num_features: int, dtype: jnp.dtype, rngs):
        super().__init__()
        self.rmsnorm = flax.nnx.RMSNorm(
            num_features=num_features, dtype=dtype, rngs=rngs
        )

      def __call__(self, x) -> jax.Array:
        return self.rmsnorm(x)

    self._run_model_tests(
        "nnx.RMSNorm",
        config,
        lambda c: Model(
            c.num_features, pt2jax_dtype(c.dtype), flax.nnx.Rngs(0)
        ),
        lambda c, key: jax.random.normal(
            key,
            (c.batch_size, c.seq_len, c.num_features),
            dtype=pt2jax_dtype(c.dtype),
        ),
        is_jax=True,
    )

  # Scaled dot product attention tests
  # ############################################################################

  _sdpa_configs_with_math_backend = ops_test_data.SDPA_CONFIGS + [
      replace(config, use_math_backend=True)
      for config in ops_test_data.SDPA_CONFIGS
  ]

  @parameterized.named_parameters(
      ops_test_data.generate_configs_for_parameterized(
          _sdpa_configs_with_math_backend
      )
  )
  def test_nn_Sdpa(self, config):
    """Benchmark torch.nn.functional.scaled_dot_product_attention."""
    if not is_pytorch_framework():
      self.skipTest("PyTorch not enabled")

    class AttentionLayer(torch.nn.Module):

      def __init__(self, c: ops_test_data.SdpaConfig):
        super().__init__()
        self.c = c
        self.q_proj = torch.nn.Linear(
            self.c.embed_dim,
            self.c.q_num_heads * self.c.qk_head_dim,
            bias=False,
            dtype=c.dtype,
        )
        self.k_proj = torch.nn.Linear(
            self.c.embed_dim,
            self.c.kv_num_heads * self.c.qk_head_dim,
            bias=False,
            dtype=c.dtype,
        )
        self.v_proj = torch.nn.Linear(
            self.c.embed_dim,
            self.c.kv_num_heads * self.c.v_head_dim,
            bias=False,
            dtype=c.dtype,
        )
        self.out_proj = torch.nn.Linear(
            self.c.q_num_heads * self.c.v_head_dim,
            self.c.embed_dim,
            bias=False,
            dtype=c.dtype,
        )

      def forward(self, x) -> torch.Tensor:
        bsz, q_len, _ = x.size()
        q = (
            self.q_proj(x)
            .view(bsz, q_len, self.c.q_num_heads, self.c.qk_head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(bsz, q_len, self.c.kv_num_heads, self.c.qk_head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(bsz, q_len, self.c.kv_num_heads, self.c.v_head_dim)
            .transpose(1, 2)
        )

        with attention.sdpa_kernel(
            [attention.SDPBackend.MATH]
            if self.c.use_math_backend
            else [attention.SDPBackend.OVERRIDEABLE]
        ):
          attn_output = torch.nn.functional.scaled_dot_product_attention(
              q, k, v, is_causal=self.c.is_causal, enable_gqa=self.c.enable_gqa
          )
        attn_output = attn_output.transpose(1, 2).reshape(
            bsz, q_len, self.c.q_num_heads * self.c.v_head_dim
        )
        return self.out_proj(attn_output)

    self._run_model_tests(
        "nn.f.scaled_dot_product_attention",
        config,
        lambda c: AttentionLayer(c),
        lambda c: torch.randn(
            (c.batch_size, c.q_seq_len, c.embed_dim),
            dtype=c.dtype,
            device=get_torch_device(),
        ),
        skip_bw_pass=(
            is_pytorch_framework() and _DEVICE.value == "cuda"
        ),  # The BW pass fails on PT/CUDA
    )

  @parameterized.named_parameters(
      ops_test_data.generate_configs_for_parameterized(
          ops_test_data.SDPA_CONFIGS
      )
  )
  def test_nnx_Sdpa(self, config):
    """Benchmark jax.nn.dot_product_attention."""
    if not is_jax_framework():
      self.skipTest("JAX not enabled")

    class AttentionLayer(flax.nnx.Module):

      def __init__(self, c: ops_test_data.SdpaConfig, rngs):
        super().__init__()
        self.c = c
        dtype = pt2jax_dtype(c.dtype)
        self.q_proj = flax.nnx.Linear(
            self.c.embed_dim,
            self.c.q_num_heads * self.c.qk_head_dim,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )
        self.k_proj = flax.nnx.Linear(
            self.c.embed_dim,
            self.c.kv_num_heads * self.c.qk_head_dim,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )
        self.v_proj = flax.nnx.Linear(
            self.c.embed_dim,
            self.c.kv_num_heads * self.c.v_head_dim,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )
        self.out_proj = flax.nnx.Linear(
            self.c.q_num_heads * self.c.v_head_dim,
            self.c.embed_dim,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )

      def __call__(self, x) -> jax.Array:
        bsz, q_len, _ = x.shape
        q = (
            self.q_proj(x)
            .reshape(bsz, q_len, self.c.q_num_heads, self.c.qk_head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(bsz, q_len, self.c.kv_num_heads, self.c.qk_head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(bsz, q_len, self.c.kv_num_heads, self.c.v_head_dim)
            .transpose(0, 2, 1, 3)
        )

        attn_output = jax.nn.dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, q_len, self.c.q_num_heads * self.c.v_head_dim
        )
        return self.out_proj(attn_output)

    self._run_model_tests(
        "nnx.f.dot_product_attention",
        config,
        lambda c: AttentionLayer(c, flax.nnx.Rngs(0)),
        lambda c, key: jax.random.normal(
            key,
            (c.batch_size, c.q_seq_len, c.embed_dim),
            dtype=pt2jax_dtype(c.dtype),
        ),
        is_jax=True,
    )

  # ############################################################################
  # Nonzero operator tests
  # ############################################################################

  @dataclass
  class _NonzeroConfig:
    shape: tuple[int, ...]
    dtype: torch.dtype
    sparsity: float

  _nonzero_configs = (
      # Default config for smoke test.
      _NonzeroConfig(
          shape=(128, 128),
          dtype=torch.bfloat16,
          sparsity=0.1,
      ),
      # Larger configs.
      _NonzeroConfig(
          shape=(8192, 8192),
          dtype=torch.bfloat16,
          sparsity=0.01,
      ),
  )

  @parameterized.named_parameters(
      generate_configs_for_parameterized(_nonzero_configs)
  )
  def test_nn_nonzero(self, config):
    """Benchmark nn.nonzero with random data."""
    if not is_pytorch_framework():
      self.skipTest("PyTorch not enabled")

    class Model(torch.nn.Module):

      def forward(self, x) -> torch.Tensor:
        return torch.nonzero(x)

    def inputs_builder(c):
      x = torch.randn(c.shape, dtype=c.dtype, device=get_torch_device())
      mask = (torch.rand(c.shape, device=get_torch_device()) < c.sparsity).to(
          c.dtype
      )
      return x * mask

    self._run_model_tests(
        "nn.nonzero",
        config,
        lambda c: Model(),
        inputs_builder,
        skip_bw_pass=True,
    )

  @parameterized.named_parameters(
      generate_configs_for_parameterized(_nonzero_configs)
  )
  def test_nnx_nonzero(self, config):
    """Benchmark jnp.nonzero with random data."""
    if not is_jax_framework():
      self.skipTest("JAX not enabled")

    class Model(flax.nnx.Module):

      def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.max_size = 1
        for s in shape:
          self.max_size *= s

      def __call__(self, x) -> jax.Array:
        return jnp.stack(jnp.nonzero(x, size=self.max_size), axis=-1)

    def inputs_builder(c, key):
      x = jax.random.normal(key, c.shape, dtype=pt2jax_dtype(c.dtype))
      mask = (jax.random.uniform(key, c.shape) < c.sparsity).astype(
          pt2jax_dtype(c.dtype)
      )
      return x * mask

    self._run_model_tests(
        "nnx.nonzero",
        config,
        lambda c: Model(c.shape),
        inputs_builder,
        skip_bw_pass=True,
        is_jax=True,
    )

  # ############################################################################
  # Topk operator tests
  # ############################################################################

  @dataclass
  class _TopkConfig:
    shape: tuple[int, ...]
    k: int
    dtype: torch.dtype
    dim: int = -1

  _topk_configs = (
      # Default config for smoke test.
      _TopkConfig(
          shape=(128, 16),
          k=2,
          dtype=torch.bfloat16,
      ),
      # Larger configs.
      _TopkConfig(
          shape=(8192, 256),
          k=8,
          dtype=torch.bfloat16,
      ),
  )

  @parameterized.named_parameters(
      generate_configs_for_parameterized(_topk_configs)
  )
  def test_nn_topk(self, config):
    """Benchmark torch.topk with random data."""
    if not is_pytorch_framework():
      self.skipTest("PyTorch not enabled")

    class Model(torch.nn.Module):

      def __init__(self, k: int, dim: int):
        super().__init__()
        self.k = k
        self.dim = dim

      def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.topk(x, self.k, dim=self.dim)

    self._run_model_tests(
        "nn.topk",
        config,
        lambda c: Model(c.k, c.dim),
        lambda c: torch.randn(
            c.shape, dtype=c.dtype, device=get_torch_device()
        ),
        skip_bw_pass=True,
    )

  @parameterized.named_parameters(
      generate_configs_for_parameterized(_topk_configs)
  )
  def test_nnx_topk(self, config):
    """Benchmark jax.lax.top_k with random data."""
    if not is_jax_framework():
      self.skipTest("JAX not enabled")

    class Model(flax.nnx.Module):

      def __init__(self, k: int, axis: int):
        super().__init__()
        self.k = k
        self.axis = axis

      def __call__(self, x) -> tuple[jax.Array, jax.Array]:
        # jax.lax.top_k always operates on the last dimension.
        return jax.lax.top_k(x, self.k, axis=self.axis)

    self._run_model_tests(
        "nnx.topk",
        config,
        lambda c: Model(c.k, c.dim),
        lambda c, key: jax.random.normal(
            key, c.shape, dtype=pt2jax_dtype(c.dtype)
        ),
        skip_bw_pass=True,
        is_jax=True,
    )


if __name__ == "__main__":
  absltest.main()
